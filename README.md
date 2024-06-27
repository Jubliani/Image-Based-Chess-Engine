# Image-Based Chess Engine
 An experiement to see how well a chess engine trained purely on images would perform.<br>
 
 So far, the engine will only consider if the position is "good", "bad" or "neutral" from white's perspective, assuming it is white to play. Training images were sourced from the Lichess open data base for July 2022. For the sake of training, a "good" and "bad" position were those evaluated to be at least 1.5 in white/black's favor in the database (and hence "neutral" being inbetween).<br>

Why did I do this? In the wise words of Cave Johnson:<br>

<b><p style="font-style: italic;">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"Science isn't about WHY. It's about WHY NOT."
</p></b>
 # Testing the models
 To try out a model, simply run the chessEngine/modelInteractive.py file and follow the prompts
 # Results so far
 Here are plots of the models I've trained. The goal of the project was just to see if this even slightly works, so decisions regarding hyperparameters and types of models were somehwat arbitrary (you can view these images in the chessEngine/results folder).<br>

 **Neural networks**<br>
 
 <img src="./chessEngine/results/EfficientNetB0_B32_E5_Lr1e-3_Wd0.png" width="500">  <img src="./chessEngine/results/EfficientNetB0_B64_E5_Lr1e-3_Wd0.png" width="500"><br>
 <img src="./chessEngine/results/EfficientNetB0_B32_E50_Lr1e-3_Wd0.png" width="500">  <img src="./chessEngine/results/EfficientNetB0_B32_E20_Lr1e-3_Wd1e-4.png" width="500"><br>


 ***TODO:*** try using an SVM instead?
