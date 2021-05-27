# Hunter × Hunter Character Image Search
## *Prerequisite Knowledge: Machine Learning, Python, Vector Math*
---
This is a step by step demonstration of how I build a Hunter × Hunter character image search application using machine learning. Basically a program that takes an image as an query and returns results that classify the image (Similar to what Google reverse image search does). The most common way to implement this would be to create a image classfication model but instead I will be implementing this with a ranking machine learning model.

---

## Background
### Ranking vs Classification
If you have spent any time looking into machine learning then there is a good chance that you know what a classification model does. A classification model takes input data proccesses it and then spits out what class it is. A classification model would be able to perform image search but a classifcation model is limited to the classes it was trained on. So the model would perform poorly on characters it has never seen before.

Ranking is taking query, image data in our case, and then comparing it to every data entry in a data collection and ranking/sorting the data entry (i.e. Google takes a text query and then ranks how relevant websites on the internet are compared to the text query). A ranking model learns to compare data so a ranking model is not implicity bound to a class set.

### Hunter × Hunter
An action adventure anime that I really like :). This show has a lot of unique characters which is why I chose it as my image data domain for image searching.

---

## Data
The data set is composed of images of 72 Hunter × Hunter characters with each character having 25 images. The dataset has 1800 images in total and is 578.2 MB in size. I got the images from Google image search. 

## Implementation
### Design
I am building a machine learning model that needs to learn how to rank images of characters from Hunter × Hunter and to do that we need to commpare images. Because of dimensionality it would be computational difficult to compare the images directly so I will use a technique called embedding. The machine learning model will "embed" images by taking an image and processing it and giving a vector as output. The vector representation of these can be compared with vector math. Cosine similarity can be used to compare how close vectors are to each other.

The machine learning model will be trained using a Triplet Loss. A triplet loss takes 3 data entries which will be denoted as Anchor, Positive, & Negative. The Anchor is an any character image in the dataset; the Positive is an image that is the same character as the Anchor; and the Negative is an image of a character that is different to the Anchor. The triplet loss measures how well the model is at making embedding representations that have the embddings of the Anchor and Positive close together and the Anchor and Negative far apart. So the machine learning during training is moving embedding representaitons images of the same character closer together and those of the different characters farther apart.

Once the machine learning model is created and trained we can take any image of a Hunter × Hunter character and get a vector that well represents it meaning that different images of the character should give vectors that are very close to each other. Since we are using a ranking algorithm we need a data collection to compare our query to. This data collection will contain all the vector representation of all the images in the dataset and then a query vector will compare itself to the vectors in the data collection.

Ranking requires us to compare our query to a large data collection which can be computationally difficult since the time complexity of comparing a query to a data collection is O(n) while using a classification model would be O(1). To speed up the query search up a Approximate Nearest Neighbors algorithm will be used. This algorithm speeds up query search at the cost of slight inaccuracies. 


### Code
I used python and tensorflow to create the machine learning model. I combined a trainable model and a non trainable model with imagenet trained weights to use transfer learning to speed up training. The loss used is TripletSemiHardLoss. To store the vector representation I used the [Annoy python library](https://github.com/spotify/annoy) which stores vectors and queries them using the Approximate Nearest Neighbors algorithm.

Model Code:

```python
dimension = 16
height = 64
width = 64
learning_rate = 5e-5

inputs = layers.Input(shape=(height, width, 3))

base_model = MobileNetV2(input_shape=(height, width, 3), include_top=False, weights='imagenet')
x = base_model(inputs)
base_model.trainable = False
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(units=1280, activation="relu")(x)

train_model = MobileNetV2(input_shape=(height, width, 3), include_top=False)
train_model._name = "training_mobilenet"
train_model = train_model(inputs)
train_model = layers.GlobalAveragePooling2D()(train_model)
train_model = layers.Dense(units=1280, activation="relu")(train_model)

x = layers.Concatenate(axis=-1)([train_model, x])
x = layers.Dense(units=2560, activation="relu")(x)
embeddings = layers.Dense(units=dimension, activation=None)(x)
embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

model = EmbeddingModel(inputs, embeddings)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tfa.losses.TripletSemiHardLoss(),
)
```

### Training
Hyperparameters:
* dimension: 16
* height: 64
* width: 64
* epoch: 200
* batch_size: 64
* learning_rate: 5e-5

The hyperparameters was tuned by hand.

GPU: NVIDIA GeForce RTX™ 30 SERIES

## Results

| Query Image (Not in Dataset) | Most similar results (Correct result in bold) |
| ----------- | ----------- |
| Killua ![Image of Killua](https://p.favim.com/orig/2019/01/27/hunter-x-hunter-killua-anime-Favim.com-6821948.png) | **Killua Zoldyck**, Pariston Hill, Ging Freecss, Melody, Pakunoda, Shalnark, Zeno Zoldyck, Gon Freecss, Baise, Cheetu|
| Chimera Ant Queen ![Image of Ant Queen](https://i.ytimg.com/vi/FW0iVENflAE/mqdefault.jpg) | Shaiapouf, Rammot, Milluki Zoldyck, **Chimera Ant Queen**, Meruem, Feitan Portor, Komugi, Chrollo Lucilfer, Shalnark, Leol|
| Cheetu ![Image of Cheetu](https://i.pinimg.com/originals/f1/64/f8/f164f886d4a0c3a4461460d62cc707d7.jpg) | **Cheetu**, Mito, Killua Zoldyck, Kurapika, Neon Nostrade, Kikyo Zoldyc, Kanzai,Shalnark, Gon Freecss, Kite |
| Kite ![Image of Kite](https://ami.animecharactersdatabase.com/uploads/chars/thumbs/200/5688-2045549894.jpg) | **Kite**, Knov, Feitan Portor, Nobunaga Hazama, Neferpitou, Morel Mackernasey, Welfin, Leol, Leorio Paradinight, Chimera Ant Queen |
| Hisoka ![Image of Hisoka](https://miro.medium.com/max/3840/1*VWzhICRDggGCgNBdLvSNAA.png) | Pokkle, Neon Nostrade, Mito,Menthuthuyoupi', Shaiapouf, Hunter Ponzu, Genthru, Machi Komacine, Cheadle Yorkshire, Squala ... **Hisoka Morow** (14th rank)' |
| Hisoka Cosplay ![Image of Hisoka Cosplay](https://i.pinimg.com/originals/2e/a3/63/2ea363d67b2d9376a5929146f762fe25.jpg) | **Hisoka Morow**, Mito, Squala, Meleoron, Ponzu, Alluka Zoldyck, Illumi Zoldyck, Nobunaga Hazama, Tonpa, Genthru |
| Meurem Fanart ![Image of Meurem fan art](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRANKshU2abrgkyRfQN67Oza-59gyVvmpq93w&usqp=CAU') | **Meruem**, Zeno Zoldyck, Pariston Hill, Saccho Kobayakawa, Hunter Leol, Kanzai, Cheetu, Isaac Netero, Basho, Colt |

### Analysis
The model did a pretty good job of classifying the images. But the model is not very robust since sometimes it fails to classify an image in the top 10 rank. 

Future improvements:
* Increase dataset size
* Clean dataset for false positive images
