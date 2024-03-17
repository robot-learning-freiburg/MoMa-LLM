# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.
import torch
from sentence_transformers import SentenceTransformer


class SentenceBERT:
    def __init__(self,):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    def get_embeddings(self, sentences):
        embeddings = self.model.encode(sentences)
        return embeddings

    def compute_cooccurrence_bipartite(self, set_1, set_2):
        if isinstance(set_1, str):
            set_1 = [set_1]
        if isinstance(set_2, str):
            set_2 = [set_2]

        sentences = set_1 + set_2
        #Sentences are encoded by calling model.encode()
        embeddings = self.get_embeddings(sentences)
        embedding_set_1 = torch.from_numpy(embeddings[0:len(set_1)])
        embedding_set_2 = torch.from_numpy(embeddings[len(set_1):])

        cooccur = torch.zeros((len(set_1), len(set_2)))
        for i in range(len(set_1)):
            for j in range(len(set_2)):
                cooccur[i, j] = self.cos_sim(embedding_set_1[i], embedding_set_2[j])
        return cooccur.detach().numpy()

    def compute_cooccurrence_unipartite(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences] 
        embeddings = torch.from_numpy(self.get_embeddings(sentences))
        cooccur = torch.zeros((len(sentences), len(sentences)))
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                cooccur[i, j] = self.cos_sim(embeddings[i], embeddings[j])
        return cooccur.detach().numpy()



if __name__ == "__main__":
    #Our sentences we like to encode
    tasks = ['Find the red towel',
                    'Can you sharpen the pencil',
                    'I want to watch a crime series. Turn on the device.']

    frontiers = ['toilet, shower, sink',
                        'fridge, oven, microwave',
                        'coffeetable, couch, TV',
                        'bed, nightstand, lamp',
                        'desk, chair, computer']
    
    sbert = SentenceBERT
    cooccur = sbert.compute_cooccurrence_bipartite(tasks, frontiers)

    import matplotlib.pyplot as plt
    plt.imshow(cooccur.detach().numpy())
    plt.show()