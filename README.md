## Vision
Text classification problems are a mainstay in machine learning, but many are done with relatively boring datasets by my standards. While plenty of useful information can be extracted from movie or product reviews, the potential classification from these did not seem very interesting. I love religious writings because understanding a person’s metaphysics is important to knowing them well. I was wondering what classification could be done with religious texts and the most obvious seemed to be if a neural network could determine the religious affiliation of a piece of text. I see this as an interesting problem mostly for the muddled examples. For instance, American thought (including religious) is being more and more influenced by an Eastern worldview including their religious outlooks. Could a neural network find these influences? For instance, could an article on Christianity containing strong buddhist ideas or themes be detected? I was probably naive to think that could be constructed within the scope of this project, but I was able to produce a neutral network that attempts to find religious affiliation. 

## Running the Code
After cloning the repo, navigate into the folder and run:
`pip3 install -r requirements.txt`

To run the NN:
`python3 prediction.py`

## Structure
The main script for the project is prediction.py.
Preproccessing of the text is handled in the proceesing/ folder, specifically in data_prep.py and scrape_folder.py.
There is one class that acts as a temporary dictionary of sorts, mapping a text to its label. It is found in the classes folder.
trainer.py has the code which trains and tests the model.
