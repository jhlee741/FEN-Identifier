# FEN-Identifier

# Introduction
The goal of this project is to generate Forsyth-Edwards Notation (FEN) descriptions for a particular chess game from a picture of a 2D-board. This notation defines the board in one line of ASCII characters (for more information click the following: FEN). This project has implications on the chess community because it creates an efficient, cost-effective way of storing, sending and analysing chessboard data allowing two parties to quickly set up board positions to analyze and learn from instead of sending/storing images. 

Machine learning is a reasonable approach for this project as it involves multi-classification of 12 chess pieces and properly outputting FEN. With JPEG files, hard coding FEN notation wouldn’t work; automation would be easier and more efficient with ML instead of human manipulation. The motivation behind the project was based on an overall team interest in chess and how to effectively store the chessboard data (FEN). With the popularization of online chess, we decided to focus on 2D boards. Current chess software stores the FEN already, but without this attribute or from a given image, this task is not readily available.

# Workflow
<img width="739" alt="Screen Shot 2021-09-01 at 2 02 07 PM" src="https://user-images.githubusercontent.com/75803498/131720960-4fd2c284-958f-4ab9-8bf8-65b269999bb3.png">

# Data Processing
***Source of Data***

We will be using the “Chess Positions” dataset from Kaggle to train our model. The author (Pavel Koryakin) generated the dataset using a custom-made chess-generator tool that randomly generates pieces on a chess board, varying in style (28 board styles, 32 chess piece styles taken from Chess.com) and number of pieces (5-15 pieces: 2 kings and 3-13 pawns/pieces). The train and test set uses different styled chess boards and pieces.

This dataset (400x400 pixels each, JPEG format) has 80,000 training images and 20,000 test images. Each image is labeled in FEN, using dashes instead of slashes in the file name. There are some images within this dataset that have illegal positions where both kings are under check; this does not concern us because analyzing possible moves is not within the scope of our project.

***Biases***

We are only using electronic 2D chess images; we do not have 3D or real life images of chess boards. This can create a bias of being able to recognize chess pieces. Additionally, each board in our dataset only consists of 5-15 chess pieces. The probability distribution of the pieces of the randomly generated images is 30% for Pawn, 20% for Bishop, 20% for Knight, 20% for Rook, 10% for Queen, with 2 Kings guaranteed on the board. It is unclear how the author chose to set these parameters, thus creating a bias of potentially having more images with certain pieces. 

***Assumptions***

The number of boards and chess styles are sufficient enough for our model to generalize to styles never seen before.

***Procedure***

<img width="607" alt="Screen Shot 2021-09-01 at 1 58 45 PM" src="https://user-images.githubusercontent.com/75803498/131720565-2f1be9fb-4e34-4d3c-8678-5581830a6eb7.png">

1. Create a Python script to extract and clean JPEG file names because FEN notation uses slashes and not dashes. This will represent the true label of the given image.
2. Shrink images to 200x200 pixels using Python script.
3. Convert each image into a numpy array (array will represent each pixel within the given image) using the Pillow or Matplotlib library.
4. Combine the information extracted in steps 1 and 2 and write it to a csv file. This information will be used for training, validation, and testing.

# Architecture

The final model used for our results is YOLOv5 (a variation of ‘You Only Look Once’ object detection model, pre-trained on COCO dataset). This model can be accessed through Pytorch and at the GitHub repository made by Ultralytics. YOLOv5 has 9 different variations, each variation differing in the number of parameters and expected image resolution. Due to the computation limit of Google Colab and image resolution of our dataset, we decided to utilize YOLOv5x which has ~87 million parameters. The YOLOv5 model is made of a backbone (trained on CSPDarknet) that does feature extraction, a neck (PANet) that does feature fusion, and a head (YOLO layer) that does the detection. YOLOv5 uses various convolutional layers, max/average pooling, bottlenecks, SPP, activation functions (LeakyReLU, SiLU, Mish, Swish), concatenations, and upsamplings. 

Using a custom dataset, we trained the YOLOv5 model to identify all instances of a chess piece in a given image on a chess board. By using a SGD optimizer, a batch size of 15, and epoch of 100 (saving the best model out of 100 epochs), the model was able to achieve a 90% accuracy on the test set. The confidence and IOU (intersection over union) thresholds were set to 0.5. Refer to Appendix A for Google Colab notebooks for model code.

An object detection model was used instead of a regular classification model because we needed a way to locate the exact position of the chess piece detected on the board. Through a regular convolution model, we would have to teach the model to identify the grids on the board, and the infinite combinations of chess positions. Thus, it seemed much easier to use an object detection model and use the bounding boxes predicted to determine the position of the piece on the board instead; with the assumption that the image given to the model is of only the chess board with pieces (2D) in their respective squares. 

# Quantitative Results

Overall the results of our model were surprisingly good. In our Precision plot, only the black rook had a low spike in precision, but generally, precision was at 1.00 precision at 0.845 confidence

<img width="371" alt="Screen Shot 2021-09-01 at 2 04 43 PM" src="https://user-images.githubusercontent.com/75803498/131721299-15b292dc-eb13-4d14-94c2-d9bf0bc940a2.png">

***Confusion Matrix***

<img width="442" alt="Screen Shot 2021-09-01 at 2 06 28 PM" src="https://user-images.githubusercontent.com/75803498/131721502-2f28efaf-1e2f-4e62-a229-76ac875baa30.png">

The main goal was to correctly produce FEN and our model did so with **~90%** accuracy from a test set of 61. The errors were exactly the issues outlined in our confusion matrix where kings and queens flip flopped and rooks were mistaken for knights and the opposite colour rook.

# Example
| Input | Output|
| :---:   | :-: |
| <img width="249" alt="Screen Shot 2021-09-01 at 2 09 37 PM" src="https://user-images.githubusercontent.com/75803498/131721907-248fed10-0460-4ea7-9430-4797a246491c.png"> | <img width="302" alt="Screen Shot 2021-09-01 at 2 09 54 PM" src="https://user-images.githubusercontent.com/75803498/131721964-d8022b83-60d4-47bc-9414-b3e798efa276.png">|
