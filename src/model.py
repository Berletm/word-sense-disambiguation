from sentence_transformers import SentenceTransformer
from ruwordnet import RuWordNet
from pymorphy3 import MorphAnalyzer
from constants import DEFAULT_ENCODER_MODEL
from typing import Tuple, List

class Lesk:
    def __init__(self, encoder_model: str = DEFAULT_ENCODER_MODEL, metric: str = "cosine"):
        self.sentence_encoder: SentenceTransformer = SentenceTransformer(encoder_model, 
                                                                         device="cuda", 
                                                                         similarity_fn_name=metric)
        self.thesaurus: RuWordNet = RuWordNet()
        self.morph: MorphAnalyzer = MorphAnalyzer(lang="ru")
        
    def disambiguate(self, context: str, target: str) -> Tuple[List]:
        context_embedding = self.sentence_encoder.encode(context)
        target_normalized = self.morph.parse(target)[0].normal_form
        senses = self.thesaurus.get_senses(target_normalized)
        
        extended_senses = []
        for sense in senses:
            extended_sense = sense.synset.title + f" ".join([h.title for h in sense.synset.hypernyms])
            extended_senses.append((extended_sense, sense.synset_id))
        
        sense_embeddings = [(self.sentence_encoder.encode(sense), i) for sense, i in extended_senses]
        
        scores = [(self.sentence_encoder.similarity(context_embedding, sense_embedding), i) for sense_embedding, i in sense_embeddings]

        max_score, synset_id = max(scores, key=lambda x: x[0])
        
        synset = self.thesaurus.get_synset_by_id(synset_id)
        
        synonims   = [synonym.name for synonym in synset.senses]
        hyponims   = [hyponim.title for hyponim in synset.hyponyms]
        hyperonims = [hypernyms.title for hypernyms in synset.hypernyms]
        
        return max_score, synonims, hyponims, hyperonims
        
    def __call__(self, context: str, target: str) -> Tuple[List]:
        return self.disambiguate(context, target)