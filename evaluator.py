import librosa
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import matplotlib.pyplot as plt

import seaborn as sn

import accent_model
import speaker_model
import inference_alon
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
# from pandas_ml import ConfusionMatrix

from tqdm import tqdm


class Evaluator:
    def __init__(self, speakers_list):
        ## accent model section
        self.accent_m = accent_model.AccentModel()
        self.accent_mapping_dict = self.accent_m.get_accent_mapping()
        self.accent_to_id_dict = {v: k for k, v in self.accent_mapping_dict.items()}

        self.accent_list = list(self.accent_to_id_dict.keys())

        ## wev2vec2 section
        self.processor_wav2vec2 = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model_wav2vec2l = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

        ## speaker model section
        self.speaker_m = speaker_model.SpeakerModel()

        self.speakers_list = speakers_list

    def evaluate_accent(self, waveform_path, final_prediction=True):
        waveform, Librosa_sample_rate = librosa.load(waveform_path, sr=16_000)
        output_distribution = self.accent_m(torch.Tensor(np.expand_dims(waveform, axis=0)))
        if final_prediction:
            output_accent = self.accent_mapping_dict[np.argmax(output_distribution)]
        else:
            output_accent = output_distribution
        return output_accent

    def evaluate_speaker(self, waveform_path, final_prediction=True):
        waveform, Librosa_sample_rate = librosa.load(waveform_path, sr=16_000)
        output_distribution = self.speaker_m(torch.Tensor(waveform).unsqueeze(0))
        if final_prediction:
            output_speaker = self.speakers_list[np.argmax(output_distribution)]
        else:
            output_speaker = output_distribution
        return output_speaker

    def evaluate_transcription(self, waveform_path):
        waveform, Librosa_sample_rate = librosa.load(waveform_path, sr=16_000)
        inputs = self.processor_wav2vec2(waveform, sampling_rate=Librosa_sample_rate, return_tensors="pt")
        with torch.no_grad():
            logits = self.model_wav2vec2l(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor_wav2vec2.batch_decode(predicted_ids)
        return transcription[0]


def get_speaker_and_accent_confusion_matrix(data, accent_list, speakers_list, output_name):
    plt.figure()
    y_actu_accent = data['base_accent']
    y_pred_accent = data['classified_accent']
    accent_confusion_matrix = get_confusion_matrix(y_actu_accent, y_pred_accent)
    svm = sn.heatmap(accent_confusion_matrix, annot=False)
    plt.title(f'{output_name} accent confusion matrix')
    figure = svm.get_figure()
    figure.savefig(f'evaloator_outputs/{output_name}_accent_confusion_matrix.png', dpi=2000)

    accent_data = []
    for accent_name in accent_list:
        if accent_name not in accent_confusion_matrix.keys():
            print(f'there is no {accent_name} in the results')
            res_data = {'accent_name': accent_name, 'accuracy': 0.0, 'recall': 0.0}
        else:
            # calc_accuracy_and_recall_per_index(accent_confusion_matrix, 'American')
            accuracy, recall = calc_accuracy_and_recall_per_index(accent_confusion_matrix, accent_name)
            res_data = {'accent_name': accent_name, 'accuracy': accuracy, 'recall': recall}
        accent_data.append(res_data)
    accent_df_results = pd.DataFrame(accent_data)
    mean_accuracy = accent_df_results.accuracy.mean()
    mean_recall = accent_df_results.recall.mean()
    res_data = {'accent_name': 'mean', 'accuracy': mean_accuracy, 'recall': mean_recall}

    accent_df_results.append(res_data, ignore_index=True)

    accent_df_results.to_csv(f'evaloator_outputs/{output_name}_accent_results.csv')


    plt.figure()
    y_actu_speaker = data['base_speaker']
    y_pred_speaker = data['classified_speaker']
    speaker_confusion_matrix = get_confusion_matrix(y_actu_speaker, y_pred_speaker)
    svm = sn.heatmap(speaker_confusion_matrix, annot=False)
    plt.title(f'{output_name} speaker confusion matrix')
    figure = svm.get_figure()
    figure.savefig(f'evaloator_outputs/{output_name}_speaker_confusion_matrix.png', dpi=2000)


    speaker_data = []
    for speaker_name in speakers_list:
        if speaker_name not in speaker_confusion_matrix.keys():
            print(f'there is no {speaker_name} in the results')
            res_data = {'speaker_name': speaker_name, 'accuracy': 0.0, 'recall': 0.0}
        else:
            # calc_accuracy_and_recall_per_index(speaker_confusion_matrix, 'American')
            accuracy, recall = calc_accuracy_and_recall_per_index(speaker_confusion_matrix, speaker_name)
            res_data = {'speaker_name': speaker_name, 'accuracy': accuracy, 'recall': recall}
        speaker_data.append(res_data)
    speaker_df_results = pd.DataFrame(speaker_data)
    mean_accuracy = speaker_df_results.accuracy.mean()
    mean_recall = speaker_df_results.recall.mean()
    res_data = {'speaker_name': 'mean', 'accuracy': mean_accuracy, 'recall': mean_recall}

    speaker_df_results.append(res_data, ignore_index=True)
    speaker_df_results.to_csv(f'evaloator_outputs/{output_name}_speaker_results.csv')




def get_confusion_matrix(y_actu, y_pred, norm=False):
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=False)
    if norm:
        df_confusion = df_confusion / df_confusion.sum(axis=1)
    return df_confusion



def calc_accuracy_and_recall_per_index(confusion_matrix, chosen_index_name):
    mapping = {accent_name: index for index, accent_name in enumerate(confusion_matrix.columns)}
    numpy_confusion_matrix = confusion_matrix.to_numpy()
    chosen_index = mapping[chosen_index_name]
    accuracy = numpy_confusion_matrix[chosen_index, chosen_index] / numpy_confusion_matrix[chosen_index, :].sum()
    recall = numpy_confusion_matrix[chosen_index, chosen_index] / numpy_confusion_matrix[:, chosen_index].sum()
    return accuracy, recall


def calc_accuracy_and_recall_all_indexs(confusion_matrix):
    numerator = 0
    denominator_accuracy = 0
    denominator_recall = 0
    for chosen_index in range(12):
        numerator += confusion_matrix[chosen_index, chosen_index]
        denominator_accuracy += confusion_matrix[chosen_index, :].sum()
        denominator_recall += confusion_matrix[:, chosen_index].sum()

    accuracy = numerator / denominator_accuracy
    recall = numerator / denominator_recall
    return accuracy, recall


def calculate_stats(confusion_matrix):
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    # Overall accuracy for each class
    ACC = (TP + TN) / (TP + FP + FN + TN)
    return ACC


def main():
    generator = inference_alon.init_generator()
    speakers_list = generator.dataset.id_to_spkr
    speaker_to_dataset_index = generator.get_index_by_speaker()

    evaluator = Evaluator(speakers_list=speakers_list)
    accent_list = evaluator.accent_list

    csv_output_path = Path('evaluation_results.csv')
    if not csv_output_path.exists():
        # output_waveforms = set()
        # for uttrerance_index, (code, gt_audio, filename, _) in tqdm(enumerate(generator.dataset)):
        #     # original_speaker = generator.dataset.id_to_spkr[int(code['spkr'][0][0])]
        #     # original_accent = generator.generator.speakers_data_dict[original_speaker]['Accent']
        #
        #     # for target_speaker in speakers_list:
        #     target_speaker = generator.dataset.id_to_spkr[int(code['spkr'][0])]
        #     for target_accent in tqdm(evaluator.accent_list):
        #         # Generate waveform for all options
        #         sound_file_output_path = generator(item_index=uttrerance_index,
        #                                            target_speaker=target_speaker,
        #                                            target_accent=target_accent)
        #         output_waveforms.update(list(sound_file_output_path.values()))
        #
        #
        # output_waveforms = list(output_waveforms)
        output_dir = Path('D:\Thesis\generated_results\generated_files_accent_change_double_zero_accent_duration').glob(
            '**/*')
        output_waveforms = [x for x in output_dir if x.is_file()]

        results_list = []

        for waveform_path in tqdm(output_waveforms):
            # path_split = Path(waveform_path).stem.split('_')
            path_split = waveform_path.stem.split('_')
            base_speaker = path_split[0]
            base_accent = generator.generator.speakers_data_dict[base_speaker]['Accent']
            target_accent = path_split[-2]

            gt = True if path_split[-1] == 'gt' else False
            gen_orig = True if path_split[-1] == 'gen-orig' else False

            classified_speaker = evaluator.evaluate_speaker(waveform_path)
            classified_transcription = evaluator.evaluate_transcription(waveform_path)
            classified_accent = evaluator.evaluate_accent(waveform_path)
            data = {'waveform_path': waveform_path,
                    'base_speaker': base_speaker,
                    'base_accent': base_accent,
                    'target_accent': target_accent,
                    'classified_speaker': classified_speaker,
                    'classified_accent': classified_accent,
                    'classified_transcription': classified_transcription,
                    'ground_truth': gt,
                    'original_generated': gen_orig}
            results_list.append(data)

        df_results = pd.DataFrame(results_list)
        df_results.to_csv(csv_output_path)

    df_results = pd.read_csv(csv_output_path, index_col=0)
    gt_results = df_results[df_results['ground_truth']]
    get_speaker_and_accent_confusion_matrix(gt_results, accent_list, speakers_list, output_name='ground_truth')

    gen_orig_results = df_results[df_results['original_generated']]
    get_speaker_and_accent_confusion_matrix(gen_orig_results, accent_list, speakers_list,
                                            output_name='original_generated')

    new_generated_results = df_results[df_results['original_generated'] == False][df_results['ground_truth'] == False]
    get_speaker_and_accent_confusion_matrix(new_generated_results, accent_list, speakers_list,
                                            output_name='new_generated_results')

    print('bamm')


if __name__ == '__main__':
    main()
