import gradio as gr
from pathlib import Path
import os
import requests
import json
import pandas as pd
import ssl

import plotly.express as px
import plotly.graph_objects as go
MAPBOX_APIENDPOINT = "https://api.mapbox.com/search/geocode/v6/forward"
MAPBOX_APIKEY="pk.eyJ1Ijoic2R5NjIzIiwiYSI6ImNsbWxoeXRweDBjM2IybG8yNWczdGc0anQifQ.sijZuqzz0r7lwKpPKRaY6A"
ssl._create_default_https_context = ssl._create_unverified_context

from inference import DisasterClassifier, CustomTextClassificationPipeline

# Paths and Configuration
MODEL_PATH = Path("./model")

# Initialize the classifier

#classifier2 = DisasterClassifier(model_path)
classifier = CustomTextClassificationPipeline(MODEL_PATH)

# Define custom CSS to set the button color and the bar color
abs_path = os.path.join(os.path.dirname(__file__), "css.css")

def predict_with_threshold(text, threshold=0.5):
    predict_result = classifier.predict(text)
    classification_result = predict_result['classification']
    ner_result = predict_result.get("ner_results", None)
    '''
    results = [
        (label if prob >= threshold else f"{label} (low)", round(prob, 3))
        for label, prob in predictions
    ]
    
    '''
    if ner_result:
        geoname = []
        current_group = []
        last_index = -1
        for entity in ner_result['entities']:
            if 'entity' in entity and entity['entity'] == 'I-LOC':
                if last_index == -1 or entity['index'] == last_index + 1:
                    current_group.append(entity['word'])
                else:
                    geoname.append("".join(current_group))
                    current_group = [entity['word']]
                last_index = entity['index']
        if current_group:
            geoname.append("".join(current_group))
    else:
        geoname = ""
    print(geoname)
    print(len(geoname))
    geoname = [name.replace("▁", "") for name in geoname]

    px.set_mapbox_access_token(MAPBOX_APIKEY)
    coors = []
    for gm in geoname:
        print(gm)
        coor = requests.get(url=MAPBOX_APIENDPOINT, params={"q": gm, "access_token": MAPBOX_APIKEY})
        
        coor = coor.json()
        coors.append(coor)
    print(coors[0])
    coordinates = coors[0]["features"][0]["geometry"]["coordinates"]
    map_scatter_fig = px.scatter_mapbox(lat=[coordinates[1]], lon=[coordinates[0]])

    # map_scatter_fig.show()
    map_scatter_fig.update_layout(mapbox_style="open-street-map")
    map_scatter_fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    #map_html = map_scatter_fig.to_html(full_html=False)
    #print(map_html)
    return classification_result, ner_result, map_scatter_fig
    #return classification_result, ner_result

# Gradio interface
def launch_gradio():
    examples = [
        ["Also the tsunami warning the other day on the east coast was not a mistake or a accident Australia.. \nIt's in preparation. \
         You should know by now how all the out of sorts are automatically played down .. \nWake up Australia . Tick tock is on the"],
        ["There is a fire in the forest and it is spreading fast"],
        ["The weather is nice and sunny today"],
        ["Moderate magnitude 5.6  #Earthquake 52 miles southeast of Chiba, Japan.  10:07pm at epicenter (1h ago, 10km deep) https://t.co/1v5v9zv3Zw"]
    ]
    demo = gr.Interface(
        fn=predict_with_threshold,
        inputs=[gr.Textbox(placeholder="Enter sentence here...", label="Input Text"), gr.Slider(0, 1, value=0.5, label="Confidence Threshold")],
        outputs=[gr.Label(label="Classification Results"), gr.HighlightedText(combine_adjacent=True, label="NER Highlighted Text"), gr.Plot(label="Map")],
        examples=examples,
        css=abs_path
    )
    demo.launch()
'''
def launch_gradio():
    examples = [
        ["There is a flood in the city and it is raining heavily"],
        ["There is a fire in the forest and it is spreading fast"],
        ["The weather is nice and sunny today"],
        ["Moderate magnitude 4.1  #Earthquake 52 miles southeast of Chiba, Japan.  1:03am at epicenter (1h ago, 10km deep) https://t.co/1v5v9zv3Zw"]
    ]
    demo = gr.Interface(predict_with_threshold,
                 gr.Textbox(placeholder="Enter sentence here..."),
                 gr.HighlightedText(combine_adjacent=True),
                 examples=examples)
    demo.launch()
'''
if __name__ == "__main__":
    launch_gradio()
