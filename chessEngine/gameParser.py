import os
import random
import chess
import chess.pgn
import chess.svg
from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg

# just want to remove the chance that short games might be unable to produce a unique position
SHORT_GAME_CUTOFF = 32

class GameParser:

    def __init__(self):
        self.uniquePositions = set()

    def parseGames(self):
        gameFiles = [game for game in os.listdir("games")]

        for file in gameFiles:
            fileGame = open(f'games\{file}', 'r')
            temp = open(f'games\{file[:-4]}Modified.pgn', 'w')
            while True:
                line = fileGame.readline()
                if not line: break
                if line.startswith('1.') and '[%eval' in line:
                    temp.write(line)
            fileGame.close()
            temp.close()

    def getGameLength(self, game: chess.pgn.GameNode):
        counter = 0
        for _ in game.mainline_moves():
            counter += 1
        return counter

    def getRandomMoveNum(self, gameLength):
        randomMove = random.randrange(gameLength)
        randomMove = randomMove if randomMove % 2 == 0 else randomMove + 1
        if randomMove > gameLength:
            return randomMove - 2
        return randomMove

    def evalToLabel(self, score: str):
        if score[0] == "#":
            return "good" if score[1] == "+" else "bad"
        sign = score[0]
        if sign == "0": return "neutral"
        number = int(score[1:]) / 100
        if number >= 1.5: return "good" if sign == "+" else "bad"
        return "neutral"

    def getRandomPositionAndEvaluationFromGame(self, game: chess.pgn.GameNode, gameLength):
        while True:
            randomMoveNum = self.getRandomMoveNum(gameLength)
            board = game.board()
            evaluation = None
            for move in game.mainline():
                if randomMoveNum == 0:
                    evaluation = self.getEval(move)
                    break
                board.push(move.move)
                randomMoveNum -= 1
            if board.fen() not in self.uniquePositions and evaluation:
                self.uniquePositions.add(board.fen())
                return board.fen(), evaluation
            else:
                print("position unique? ", board.fen() in self.uniquePositions, "evaluation? ", evaluation is None)

    def createTrainingData(self):
        gameFiles = [open("games\lichess_db_standard_rated_2022-07.pgn")]
        counter = 200000
        # there's wayyy more than 200k games on here, no need for none checking
        while counter > 0:
            game = chess.pgn.read_game(gameFiles[0])
            if counter % 50 == 0:
                print("counter: ", counter)
            gameLength = self.getGameLength(game)
            if gameLength < SHORT_GAME_CUTOFF:
                continue
            fen, evaluation = self.getRandomPositionAndEvaluationFromGame(game, gameLength)
            board = chess.Board(fen=fen)
            image = chess.svg.board(board, coordinates=False, size=100,
                                    colors={"square light": "#FFFFFF", "square dark": "#555555"})

            fen = board.fen().replace("/", "_")
            with open(f'chessEngine\images\{evaluation}\output.svg', 'w') as output:
                output.write(image)
            drawing = svg2rlg(f'chessEngine\images\{evaluation}\output.svg')
            renderPM.drawToFile(drawing, f"chessEngine\images\{evaluation}\{fen}.png", fmt="PNG")
            counter -= 1
        os.remove('chessEngine\images\good\output.svg')
        os.remove('chessEngine\images\\neutral\output.svg')
        os.remove('chessEngine\images\\bad\output.svg')

    def getEval(self, move: chess.pgn.GameNode):
        evaluation = move.eval()
        if evaluation is None:
            return False
        return self.evalToLabel(str(evaluation.white()))


    # print(sum(1 for line in open('games\lichess_db_standard_rated_2022-09Modified.pgn')))
    # print(sum(1 for line in open('games\lichess_db_standard_rated_2022-09.pgn')))
    # print(sum(1 for line in open('games\sample.pgn')))
