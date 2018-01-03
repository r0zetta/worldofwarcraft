import markovify
import json
import os

markov_model = None
save_dir = "markov"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

input_file = "data/data.json"
model_save_path = os.path.join(save_dir, "markov.json")
if os.path.exists(model_save_path):
    print("Loading pre-saved model")
    with open(model_save_path, "r") as f:
        model_json = json.load(f)
        markov_model = markovify.Text.from_json(model_json)
else:
    print("Creating model from data")
    print("Loading text")
    with open(input_file, "r") as f:
        json_data = json.load(f)
        text = json_data
        print("Building model")
        markov_model = markovify.Text(text)
        print("Saving pre-trained model")
        model_json = markov_model.to_json()
        with open(model_save_path, "w") as f:
            json.dump(model_json, f, indent=4)

if markov_model is not None:
    print
    print("Randomly generated arbitrary length sentences.")
    print("==============================================")
    # Print five randomly-generated sentences
    for _ in range(20):
        print(markov_model.make_sentence())

    print
    print("Randomly generated fixed length sentences.")
    print("==========================================")

    # Print three randomly-generated sentences of no more than 140 characters
    for _ in range(20):
        print(markov_model.make_short_sentence(140))
