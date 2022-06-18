import dash
import numpy as np

from dash import Input, Output, html, dcc

import inference_alon
import accent_model
import speaker_model
import librosa
import torch

import pandas as pd
import plotly.express as px
import plotly.io as pio
from datetime import datetime
import base64
import resampy

pio.templates.default = "simple_white"

speakers_data_path = r"src/speakers_data.csv"
speakers_data = pd.read_csv(speakers_data_path).set_index('Id')
speakers_data_dict = speakers_data.to_dict('index')
speakers_list = []#list(speakers_data_dict.keys())
# accent_list = speakers_list.copy()  # speakers_data.Accent.unique().tolist()

COLOR_HEADER = "rgb(14, 88, 20)"

data_sound_file = r"D:\Thesis\generated_results\generated_files_accent_change_double_zero_accent\p226_131_mic2_speaker_p248_accent_p248_gen.wav"  # "data/Dog Snarling.mp3"
# data sound preparations
data_sound = base64.b64encode(open(data_sound_file, 'rb').read())

GENERATOR = inference_alon.init_generator('C:/git/speech-resynthesis-alon/tmp')
speakers_list = GENERATOR.dataset.id_to_spkr
speaker_to_dataset_index = GENERATOR.get_index_by_speaker()

accent_list = speakers_list.copy()

accent_m = accent_model.AccentModel()
accent_mapping_dict = accent_m.get_accent_mapping()

speaker_m = speaker_model.SpeakerModel()


# Lista con los títulos y los filtros de la barra de la izquierda.
MENU = [
    html.P("Anonymous", className="title-filter"),
    dcc.RadioItems(
        id="anonymous_button",
        options=[{"label": 'Non-Anonymous', "value": False}, {"label": 'Anonymous', "value": True}],
        value=False,
        inline=True
    ),
    html.P("Base Speaker", className="title-filter"),
    dcc.Dropdown(
        id="base_speaker_selector",
        options=[{"label": r, "value": r} for r in speakers_list],
        value=speakers_list[0],
        multi=False,
        clearable=False,
        style={"width": "100%"}
    ),
    html.P("Base Speaker Utterance index", className="title-filter"),
    dcc.Dropdown(
        id="base_speaker_index_selector",
        options=[{"label": str(r), "value": r} for r in range(5)],
        value=0,
        multi=False,
        clearable=False,
        style={"width": "100%"}
    ),
    html.P("Target Speaker", className="title-filter"),
    dcc.Dropdown(
        id="target_speaker_selector",
        options=[{"label": r, "value": r} for r in speakers_list],
        value=speakers_list[0],
        multi=False,
        clearable=False,
        style={"width": "100%"}
    ),
    html.P("Target Accent", className="title-filter"),
    dcc.Dropdown(
        id="target_accent_selector",
        options=[{"label": r, "value": r} for r in accent_list],
        value=accent_list[0],
        multi=False,
        clearable=False,
        style={"width": "100%"}
    ),
]

# GRAFICOS = [dcc.Graph(id="line-chart"), dcc.Graph(id="donut-chart")]

app = dash.Dash(__name__)
app.layout = html.Div(
    children=[
        html.Div(
            id="header",
            children=[html.H1("Generate Waveforms", id="title")],
            style={"background": COLOR_HEADER},
        ),
        html.Div(
            children=[
                html.Div(
                    id="menu",
                    children=MENU,
                ),
                # html.Div(
                #     id="graphs",
                #     children=GRAFICOS,
                # ),
                html.Audio(title='gt',
                           id='gt_player',
                           src='data:audio/mpeg;base64,{}'.format(data_sound.decode()),
                           controls=True,
                           autoPlay=False,
                           style={"width": "100%"}
                           ),
                html.Div(id='Base_Accent_results'),
                html.Div(id='Base_speaker_results'),
                html.Audio(title='gt_gen',
                           id='gt_gen_player',
                           src='data:audio/mpeg;base64,{}'.format(data_sound.decode()),
                           controls=True,
                           autoPlay=False,
                           style={"width": "100%"}
                           ),
                html.Div(id='gt_gen_Accent_results'),
                html.Div(id='gt_gen_speaker_results'),
                html.Audio(title='new_gen',
                           id='new_player',
                           src='data:audio/mpeg;base64,{}'.format(data_sound.decode()),
                           controls=True,
                           autoPlay=False,
                           style={"width": "100%"}
                           ),
                html.Div(id='new_gen_Accent_results'),
                html.Div(id='new_gen_speaker_results'),
            ],
            id="content",
        ),
    ],
    id="wrapper",
)


def get_src_from_sound_file_path(data_sound_file):
    data_sound = base64.b64encode(open(data_sound_file, 'rb').read())
    return 'data:audio/mpeg;base64,{}'.format(data_sound.decode())


# @app.callback(
#     [
#         Output("base_speaker_selector", "value"),
#         Output("target_speaker_selector", "value"),
#         Output("target_accent_selector", "value")
#     ],
#     [dash.dependencies.Input('random_button', 'n_clicks')],
# )
# def random_speakers():
#     return 'p225'
#     )

def get_accent_classification(waveform_path):
    # reading output waveforms
    waveform, Librosa_sample_rate = librosa.load(waveform_path)
    waveform = resampy.resample(waveform, Librosa_sample_rate, 16_000)
    output_accent = accent_mapping_dict[np.argmax(accent_m(torch.Tensor(np.expand_dims(waveform, axis=0))))]
    return output_accent

def get_speaker_classification(waveform_path):
    # reading output waveforms
    waveform, Librosa_sample_rate = librosa.load(waveform_path)
    waveform = resampy.resample(waveform, Librosa_sample_rate, 16_000)
    # print(f'speaker_index :{np.argmax(speaker_m(torch.Tensor(waveform).unsqueeze(0)))},{speaker_m(torch.Tensor(waveform).unsqueeze(0))}')
    output_speaker = speakers_list[np.argmax(speaker_m(torch.Tensor(waveform).unsqueeze(0)))]
    return output_speaker


@app.callback(
    [
        Output("gt_player", "src"),
        Output("gt_gen_player", "src"),
        Output("new_player", "src"),
        Output("base_speaker_index_selector", "options"),
        Output("base_speaker_selector", "options"),
        Output("target_speaker_selector", "options"),
        Output("target_accent_selector", "options"),
        Output('Base_Accent_results', "children"),
        Output('Base_speaker_results', "children"),
        Output('gt_gen_Accent_results', "children"),
        Output('gt_gen_speaker_results', "children"),
        Output('new_gen_Accent_results', "children"),
        Output('new_gen_speaker_results', "children"),
    ],
    [
        Input("anonymous_button", "value"),
        Input("base_speaker_selector", "value"),
        Input("target_speaker_selector", "value"),
        Input("target_accent_selector", "value"),
        Input("base_speaker_index_selector", "value"),
    ],
)
def update_graphs(keep_anonymous, base_speaker, target_speaker, target_accent, base_speaker_index):
    print(f'base_speaker:{base_speaker}, target_speaker:{target_speaker}, target_accent:{target_accent}')
    sound_file_output_path = inference_alon.generate_from_speakers(generator=GENERATOR,
                                                                   base_speaker=base_speaker,
                                                                   target_speaker=target_speaker,
                                                                   target_accent=target_accent,
                                                                   base_speaker_index=base_speaker_index)

    print(sound_file_output_path)

    gt_src = get_src_from_sound_file_path(sound_file_output_path['gt'])
    gt_gen_src = get_src_from_sound_file_path(sound_file_output_path['gt_gen'])
    new_src = get_src_from_sound_file_path(sound_file_output_path['new'])

    number_of_utterance = len(speaker_to_dataset_index[base_speaker])
    utterance_options = [{"label": str(r), "value": r} for r in range(number_of_utterance)]
    if keep_anonymous:
        speaker_selector = [{"label": r, "value": r} for r in speakers_list]
    else:
        speaker_selector = [{"label": f'{r}_{speakers_data_dict[r]}', "value": r} for r in speakers_list]

    gt_accent = get_accent_classification(sound_file_output_path['gt'])
    gt_gen_accent = get_accent_classification(sound_file_output_path['gt_gen'])
    new_accent = get_accent_classification(sound_file_output_path['new'])

    gt_speaker = get_speaker_classification(sound_file_output_path['gt'])
    gt_gen_speaker = get_speaker_classification(sound_file_output_path['gt_gen'])
    new_speaker = get_speaker_classification(sound_file_output_path['new'])

    return gt_src,\
           gt_gen_src,\
           new_src,\
           utterance_options,\
           speaker_selector,\
           speaker_selector,\
           speaker_selector,\
           gt_accent,\
           gt_speaker, \
           gt_gen_accent,\
           gt_gen_speaker, \
           new_accent,\
           new_speaker


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
