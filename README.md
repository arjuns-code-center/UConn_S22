# UConn_S22
Summer 2022 Research Project with Nila Mandal and Dr. Qian Yang at UConn, Storrs. Using a Siamese Neural Network to learn Eutectic Proportions/Temperatures.

# Introduction: 
Eutectic proportions and melting points prediction is a difficult problem as it usually involves very small data and extensive use of energy and time. These methods are inefficient and can often yield inconclusive results. 

To combat this, we propose a semisupervised pairwise learning method which allows learning with very small data. This method will be implemented using a Siamese Neural Network model which learns pairs of molecule properties and predicts a eutectic proportion or melting point. 

# Datasets:
Below is a picture showing the molecular features and eutectic compilation datasets. 

![image](https://user-images.githubusercontent.com/41523488/185447503-2f000913-4775-4a7e-9ce9-0c3dd1a32416.png)

At first, it was just 5 features, but these were not enough to make good, reasonable predictions. As a result, I have added some more features using PubChem. It is thee atomic number and count of each element in every molecule in molecule_features dataset, in order of least mass to most mass. With these added features, I hope to obtain better results as the model has more to learn now. 

Each feature was skew right from the boxplots (not shown), meaning there may be some outlier bias that affects the model performance. 

![image](https://user-images.githubusercontent.com/41523488/185447560-e635fc23-e816-4cf5-ad1c-d9f569147207.png)

Below are the boxplots of the Xe and Te values (labels). One thing to notice is that with Xe, there are entires with same molecule pairs but the proportions are inverted, or the proportions are very different. This is due to the dataset being constructed from multiple published sources, each arguing their own proportion. This was a confounding variable for the model performance, so we decided to train on Te. Te has lesser restrictions and the dataset seemed better prepared than Xe. 

![image](https://user-images.githubusercontent.com/41523488/185447964-529c4503-83e9-4ccc-b400-9b55aa80a59f.png)

We can see that Xe is evenly distributed, but Te has a skew right distribution. This might be a problem since the values on the far right may affect the network performance as it tries to learn. Additionally, since the features are not the best and the wide range of Te, there may be a drop of performance from that too. 

The datasets consist of 14 molecular features: 
- Molecular Weight
- Complexity
- Rotatable Bond Count
- Heavy Atom Count
- Topological Area
- Atomic numbers and counts of every element present in each molecule, added in order of least mass to most mass

All features are retreived from PubChem, a database for chemical compounds. PubChemPy, an API to access PubChem data through Python, was also used in the code to try generating molecular information for learning, though this was a separate task and proved inconclusive. 

Some facts about model and training proporties:
- Loss: MAE
- Optimizer: Adam
- Learning rate: 7.5e-5
- Train Parameter: Te
- Batchsize: 100
- Epochs: 20
- Num Features: 14

# Training: 
The Siamese Model: 

![image](https://user-images.githubusercontent.com/41523488/182658774-96e0e125-a426-48c6-a225-d613ef0a61d9.png)

Above is an image of a sample Siamese Neural Network architecture. The pair of 5 molecule features, shown in GREEN and BLACK, are connected into a Dense layer, shown in RED. Then, I compute a distance metric, shown in YELLOW, which is a partially connected layer performing a weighted difference. Finally, this is passed onto a Dense neuron which provides output. The real architecture has many more nodes, and for the sake of simplicity to promote understanding, I have included a simpler model. 

The Simple Model: 

![image](https://user-images.githubusercontent.com/41523488/182659118-60e11360-8b32-4c85-a454-d0c4ba3ce8b4.png)

Above in an image of the Simple Neural Network architecture. Both pairs of molecule features are concatenated, shown in GREEN. Then, this is passed into Dense layers shown in RED, which is passed into the final neuron in GRAY, where output is provided. Again, there are many more nodes, and for the sake of simplicity and understanding, I have shown a simpler model. 

# Results: 
![image](https://user-images.githubusercontent.com/41523488/185451300-81c73298-bff3-4c36-a640-d5f7564caeb0.png)

Above is a side by side plot of Siamese and Simple, shown on the LEFT and RIGHT respectively, for Xe. Each model has a residual and output plot. The residual plot has the residuals shown in RED, the 100% accuracy line shown in GREEN, with a tolerance lines shown in YELLOW. The output plots have the actual vs predicted points shown in GREEN, with the 100% accuracy line shown in BLUE. 

Although this seems to be good, eliminating the bad outputs may be unrepresentative of the entire model performance, and may provide unnecessary bias. It was soon found that Xe had a lot of errors in it and the model was learning things it should not have been in the first place, so we decided to abandon Xe and train on Te, since there were less restrictions for Te than there were Xe. 

![image](https://user-images.githubusercontent.com/41523488/185452119-45eab105-4d59-4ac1-9921-ed9f87c38469.png)

Above is a side by side plot of Siamese and Simple, shown on the LEFT and RIGHT respectively, for Te. Each model has a residual and output plot. The residual plot has the residuals shown in RED, the 100% accuracy line shown in GREEN. The output plots have the actual vs predicted points shown in GREEN, with the 100% accuracy line shown in BLUE. 

We can see that the predictions were very bad in both models. Many of the data points are scattered around the IQR, with nothing being predicted past the third quartile of Te boxplot. This could just be because of how the Te is distributed. There is still a lot of work to be done on the Te dataset before coming back and training again. 

# Discussion:
Given the features and skewness in labels, the models did as best as they could. Xe was found to be untrainable after investigating into its discrepancies. The Te values were very widely spread with skewness to the right. The Siamese NN could not beat the baseline, which was very concerning, and the R^2 was negative, meaning it was not able to learn anything. The Simple NN did beat the baseline, however, and had a positive, small R^2. This means it was able to learn something, but due to its weak and simple architecture, was not able to give very conclusive plots. With such a large batchsize, both networks were not able to give proper, conclusive results. This may just be due to bad features or bad labels. Once they are fixed and validated, I believe the models will do much better. This work will need to be continued in the future...
