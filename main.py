import pandas as pd
from ActiveLearner import ActiveLearner
from neural_classifier import SPECTER_CLS
from utils import add_embedding

def remit_new(simulate=False):
    mypath = "data//UK_Calls_new.csv"
    remits = ["Advanced Therapies"	,"Biomedical engineering"	,"Drug Therapy & medical device combination",	"Diagnostics",	"Artificial Intelligence",	"Gut health-microbiome-nutrition"]
    for r in remits:
        if r != "":
            print(r)
            data = pd.read_csv(mypath).fillna("")
            data = data.sample(frac=1, random_state=48)
            data.reset_index(drop=True, inplace=True)

            data["label"]=0
            data.loc[data[r] == "Y", 'label'] = 1
            if not simulate:
                print("reordering")
                print(data.shape)
                print(sum(data["label"]))

                classifier = SPECTER_CLS
                al = ActiveLearner(classifier, data, field="Summary", model_name="Neural", do_preprocess=True)
                output = al.reorder_once()
                output.rename(columns={"predictions": "{}_predictions_Summary".format(r.replace(" ", "_"))}, inplace=True)
                output.to_csv(mypath, index=False)
            else:
                print("simulating")

                data = data.drop(data[data.Include == ""].index)
                print(data.shape)
                print(sum(data["label"]))
                classifier = SPECTER_CLS
                al = ActiveLearner(classifier, data, field="Description", model_name="Neural", do_preprocess=False)
                al.simulate_learning(r)

def precompute(sents):
    from sentence_transformers import util, SentenceTransformer
    model_name = 'sentence-transformers/allenai-specter'
    model = SentenceTransformer(model_name)

    emb_source = model.encode(list(sents))  # get our data column
    cos_sim = util.pytorch_cos_sim(emb_source, emb_source)  # .diagonal().tolist()#all similarities
    return cos_sim

def sort_by_remit(simulate=False):
    mypath="H://Downloads//sonia_new.csv"
    mycol="Category"
    lbl="label"
    refcol="Tiab"
    data = pd.read_csv(mypath).fillna("")
    txts=data[refcol]
    print("precomputing embeddings")
    #cosims=precompute(txts)


    remits=data[mycol].unique()
    print(remits)
    # abstracts=[t[:900] for t in data["Description"]]
    # data["abbrev"]=abstracts
    # data.to_csv(mypath, index=False)

    for r in remits:
        if r != "":
            print(r)
            data = pd.read_csv(mypath).fillna("")
            data = data.sample(frac=1, random_state=48)
            data.reset_index(drop=True, inplace=True)

            data[lbl]=0
            data.loc[data[mycol] == r, lbl] = 1
            if not simulate:
                print("reordering")
                print(data.shape)
                print(sum(data[lbl]))

                classifier = SPECTER_CLS
                #al = ActiveLearner(classifier, data, field=refcol, model_name="Neural", do_preprocess=False, precomputed=cosims)
                al = ActiveLearner(classifier, data, field=refcol, model_name="Neural", do_preprocess=False)
                output = al.reorder_once()
                output.rename(columns={"predictions": "{}_predictions_Tiab2".format(r.replace(" ", "_"))}, inplace=True)
                output.to_csv(mypath, index=False)
            else:
                print("simulating")

                data = data.drop(data[data[mycol] == ""].index)
                print(data.shape)
                print(sum(data[lbl]))
                classifier = SPECTER_CLS
                al = ActiveLearner(classifier, data, field="Title", model_name="Neural", do_preprocess=True)
                al.simulate_learning(r)


#sort_by_remit(simulate=False)
#remit_new(simulate=True)
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




