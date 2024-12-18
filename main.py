import pandas as pd
from ActiveLearner import ActiveLearner
from neural_classifier import SPECTER_CLS
from utils import add_embedding



def precompute(sents):
    from sentence_transformers import util, SentenceTransformer
    model_name = 'sentence-transformers/allenai-specter'
    model = SentenceTransformer(model_name)

    emb_source = model.encode(list(sents))  # get our data column
    cos_sim = util.pytorch_cos_sim(emb_source, emb_source)  # .diagonal().tolist()#all similarities
    return cos_sim


#add_embedding("H://Downloads//fixed.csv", ["Abstract",	"Title"])


if __name__ == '__main__':


    data = pd.read_csv(r"C:\Users\c1049033\PycharmProjects\ncl_medx\data\full_mined_labelled.csv").fillna("")
    data = data.sample(frac=1, random_state=48)
    data.reset_index(drop=True, inplace=True)

    ##################################################NEURAL MODEL
    classifier = SPECTER_CLS
    al = ActiveLearner(classifier, data, field="ScientificTitle", model_name="Neural", do_preprocess=False)
    ##al = ActiveLearner(classifier, data, field="Interventions", model_name="Neural", do_preprocess=False)###example chosing a different field
    al.simulate_learning(plottitle="AI simulation")#ecample for simulation, can still be used to provide fancy plot to the user to see how the model would have reacted tto their data in active learning scenario

    output= al.reorder_once()
    output.to_csv("data//reordered.csv", index=False)

    ####################################################Can safely ignore this. These lines look where data was missing from the scan and supplement it with the mined data. I guess at runtime with new projects we won;t have that yet.
    # ints=data["Interventions"]
    # mined=data["mined_intervention_control"]
    # new=[ d if d != "" else mined[i] for i, d in enumerate(ints) ]#use mined data if no intervation pulled from scan. Totally optional
    # data["Interventions"]=new
    #########################################################

    ####################################Filter model
    # classifier = regexClassifier
    # al = ActiveLearner(classifier, data, field="ScientificTitle", model_name="Filter")
    # #al = ActiveLearner(classifier, data, field="Interventions", model_name="Filter")
    # al.simulate_learning()

    ################################################Random as reference
    # classifier = emptyClassifier
    # al = ActiveLearner(classifier, data, field="ScientificTitle", model_name="Random", do_preprocess=False)
    # al.simulate_learning()




