import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
import chess
import chess.pgn
import chess.svg
from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg
import os
from PIL import Image
import timm
import numpy as np
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
import pickle

class PositionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
        pass

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes
    
class PositionClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(PositionClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True, in_chans=1)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output

class InteractiveModel():

    def __init__(self):
        self.previousFEN = ""
        self.outputs = ['Bad', 'Good', 'Neutral']
    
    def FENtoPNG(self, fen):
            board = chess.Board(fen=fen)
            image = chess.svg.board(board, coordinates=False, size=100,
                                        colors={"square light": "#FFFFFF", "square dark": "#555555"})
            fen = board.fen().replace("/", "_")
            with open(os.path.join(os.getcwd(), 'interactiveImage', 'output.svg'), 'w') as output:
                output.write(image)
            svgPath = os.path.join(os.getcwd(), 'interactiveImage', 'output.svg')
            drawing = svg2rlg(svgPath)
            renderPM.drawToFile(drawing, os.path.join(os.getcwd(), 'interactiveImage', f'{fen}.png'), fmt="PNG")
            os.remove(svgPath)
            self.previousFEN = fen
    
    def deleteLastImage(self):
        os.remove(os.path.join(os.getcwd(), 'interactiveImage', f'{self.previousFEN}.png'))
    
    def EvaluateFEN(self, fen):
        pass
    
class InteractiveModelNN(InteractiveModel):

    def __init__(self, modelChoice):
        super().__init__()
        modelPath = os.path.join(os.getcwd(), 'models', 'NN', modelChoice)
        self.model = PositionClassifier()
        self.model.load_state_dict(torch.load(modelPath))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Running on device: {self.device}')
        self.model.to(self.device)
        self.model.eval()
    
    def EvaluateFEN(self, fen):
        self.FENtoPNG(fen)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Grayscale(1)
        ])
        image = transform(Image.open(os.path.join(os.getcwd(), 'interactiveImage', f'{self.previousFEN}.png'))).unsqueeze(0)
        with torch.no_grad():
            image_tensor = image.to(self.device)
            outputs = self.model(image_tensor)
        self.deleteLastImage()
        return f"Evaluation: {self.outputs[torch.argmax(outputs)]}"
    
class InteractiveModelSVM(InteractiveModel):

    def __init__(self, modelChoice):
        super().__init__()
        modelPath = os.path.join(os.getcwd(), 'models', 'SVM', modelChoice)
        with open(modelPath, 'rb') as f:
            self.model = pickle.load(f)
    
    def EvaluateFEN(self, fen):
        self.FENtoPNG(fen)
        image = Normalizer(copy=False).fit_transform([np.asarray(Image.open(os.path.join(os.getcwd(), 'interactiveImage', f'{self.previousFEN}.png')).convert('L')).flatten()])
        self.deleteLastImage()
        return f"Evaluation: {self.outputs[(self.model.predict(image))[0]]}"

def ChooseModel(modelType, modelChoice):
    if modelType == 'NN':
        return InteractiveModelNN(modelChoice)
    elif modelType == 'SVM':
        return InteractiveModelSVM(modelChoice)
    else:
        return 0