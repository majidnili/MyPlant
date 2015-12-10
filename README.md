# MyPlant - Plant Recommender System
Email: majid.nili@gmail.com

LinkedIn: https://www.linkedin.com/in/majidnili
## Description:
Finding the right plant is not easy for layman since similar looking plants can have copletely set of different needs such as light and water. Myplant uses natural language process to recommend plnats using cosine similarity between plants. 
## Data Source:
Data is scraped from www.missouribotanicalgarden.org with about 7100 well documented plant profiles. missouribotanicalgarden.org is a credible website cited by USDA.
## Modeling:
Each plant page has a tabular (structured) and body (unstructured) data parts. I converted the cathegorical features of tabular part into numerical and made few more features. I also vectorized the unstructured part using sklearn's quad-gram TfidfVectorizer and reduced the number of features with TruncatedSVD down to 150.
## Results:
The two matrices from tabular and description parts of data are concatenated and cosine similarity between the plants are calculated. The picture shows results for specific plant (Anthurium). Cosine similarity plots show that both structured (tabular) and unstructured (text) data play significant role in the final result.
![alt tag](https://raw.githubusercontent.com/majidnili/Myplant/master/images/Anthurium1.png)
## MyPlant: The webapp
In the webapp homepage user enters either botanical or common name of a plant and recieves recommenation based on

![alt tag](https://raw.githubusercontent.com/majidnili/Myplant/master/images/Anthurium2.png)

![alt tag](https://raw.githubusercontent.com/majidnili/Myplant/master/images/Homepage2.png)
