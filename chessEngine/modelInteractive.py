print("Loading...")
import torch
import modelUtils
import os
import torchvision
modelTypes = []
models = []
print(os.getcwd())
print("List of models:")
modelFolder = os.listdir('models')
for i in range(len(modelFolder)):
    modelTypes.append(modelFolder[i])
    print(f"    {i}: {modelFolder[i]}")
while True:
    try:
        modelChoice = int(input("Please select what kind of model to use by entering the folder number: "))
    except ValueError:
        print("Error: input is not an integer")
        continue
    else:
        if not (modelChoice < 0 or modelChoice >= len(modelFolder)):
            break
        print("Error: folder doesn't exist")
        continue
modelType = modelTypes[modelChoice]
modelFolder = os.listdir(os.path.join(os.getcwd(), 'models', modelType))
for i in range(len(modelFolder)):
    models.append(modelFolder[i])
    print(f"    {i}: {modelFolder[i]}")

while True:
    try:
        modelChoice = int(input("Please select your model to use by entering its number: "))
    except ValueError:
        print("Error: input is not an integer")
        continue
    else:
        if not (modelChoice < 0 or modelChoice >= len(modelFolder)):
            break
        print("Error: model doesn't exist")
        continue

interactiveModel = modelUtils.ChooseModel(modelType, modelFolder[modelChoice])
print("Successfully loaded model. To quit, type in \"q\". Good luck evaluating...\n\n")
while True:
    fen = input("Paste in FEN: ")
    if fen == 'q':
        break
    try:
        output = interactiveModel.EvaluateFEN(fen)
    except ValueError:
        print("Error: input is not a valid FEN")
        continue
    else:
        print(output)
