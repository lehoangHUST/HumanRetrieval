import numpy as np
import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Spilt labels 
'''
    Example: Image have path file ./female--Chinos-black--Tee-black_white--733.jpg
    - Split Human : female or male
    - Type clothes: [Chinos, Tee]
    - Color clothes: [black, black_white]
'''

def split_label(path: str):
  if not os.path.isfile(path):
    print("Not find path {path}")
  
  dict_clothes = {}
  # Split parent folder and file
  label = path.lower()
  label = label.split('/')[-1].replace('.jpg', '')
  label = label.split('--')
  gender = label[0]
  type_clothes = []
  color_clothes = []
  for clothes in label[1: -1]:
    typ, color = clothes.split('-')
    if typ in ['blazer', 'bomber_jacket', 'cardigan', 'tank', 'tee', 'sweater']:
        dict_clothes['top'] = [typ, color.split('_')] # Type, color clothes
    else:
        dict_clothes['bottom'] = [typ, color.split('_')] # Type, color clothes

  dict_clothes['gender'] = gender
  dict_clothes['filename'] = label[-1]
  return dict_clothes

def inv_split_label(y_true: dict):
    true_tcolor = '_'.join(y_true['top'][1])
    true_bcolor = '_'.join(y_true['bottom'][1])
    true_tclothes = '-'.join([y_true['top'][0], true_tcolor])
    true_bclothes = '-'.join([y_true['bottom'][0], true_bcolor])
    true_label = '--'.join([y_true['gender'], true_tclothes, true_bclothes, y_true['filename']])
    return true_label

def check_color_clothes(top_tcolor, top_pcolor, bottom_tcolor, bottom_pcolor):
    flags_top = []
    for _ in top_tcolor:
        if _ in top_pcolor:
            top_pcolor.pop(top_pcolor.index(_))
            flags_top.append(True)
        else:
            break
    
    flags_bottom = []
    for _ in bottom_tcolor:
        if _ in bottom_pcolor:
            bottom_pcolor.pop(bottom_pcolor.index(_))
            flags_bottom.append(True)
        else:
            break

    if len(flags_bottom) == len(bottom_tcolor) and len(flags_top) == len(top_tcolor):
        return True
    else:
        return False

def accuracy_system(y_preds: list, y_true: list):
    '''
      - y_pred: List label include [gender_true, true_type_clothes, true_color_clothes]
      - y_true: List label include [gender_pred, pred_type_clothes, pred_color_clothes]
      - accuracy: include gender, type_clothes or color_clothes 
    '''
    gender = 0
    type_clothes = 0
    color_clothes = 0
    system = 0
    list_error = {'file': [], 'error_gender': [], 'error_type': [], 'error_color': []}
    if len(y_preds) == len(y_true) and len(y_preds) != 0:
        for idx, pred  in enumerate(y_preds):
            flags = []
            # Check gender
            if pred['gender'] == y_true[idx]['gender']:
                flags.append(True)
                list_error['error_gender'].append(0)
                gender += 1
            else:
                list_error['error_gender'].append(pred['gender'])
          
            # Check type clothes 
            if pred['top'][0] == y_true[idx]['top'][0] and pred['bottom'][0] == y_true[idx]['bottom'][0]:
                flags.append(True)
                list_error['error_type'].append(0)
                type_clothes += 1
            elif pred['top'][0] != y_true[idx]['top'][0] and pred['bottom'][0] == y_true[idx]['bottom'][0]:
                list_error['error_type'].append(pred['top'][0])
            elif pred['top'][0] == y_true[idx]['top'][0] and pred['bottom'][0] != y_true[idx]['bottom'][0]:
                list_error['error_type'].append(pred['bottom'][0])
            else:
                list_error['error_type'].append(','.join([pred['top'][0], pred['bottom'][0]]))

            # Check color clothes 
            if check_color_clothes(y_true[idx]['top'][1], pred['top'][1], y_true[idx]['bottom'][1], pred['bottom'][1]):
                flags.append(True)
                list_error['error_color'].append(0)
                color_clothes += 1
            else:
                error_tcolor = '-'.join(pred['top'][1])
                error_bcolor = '-'.join(pred['bottom'][1])
                list_error['error_color'].append(','.join([error_tcolor, error_bcolor]))
            
            # System
            if len(flags) == 3:
                system += 1
            
            list_error['file'].append(inv_split_label(y_true[idx]))

        # Acc
        print(f'System : {system / len(y_preds): .4f} \t'
            f'Gender: {gender / len(y_preds): .4f} \t'
            f"Type clothes: {type_clothes / len(y_preds): .4f} \t"
            f"Color clothes: {color_clothes / len(y_preds): .4f} ")
    else:
        note = f" May be length true and perd is {len(y_preds) == len(y_true)}" 
        print(note)

    print(list_error)
    df_error = pd.DataFrame(list_error)
    df_error.to_excel('/content/error.xlsx')
    confusion_matrix_system(y_preds, y_true)

def confusion_matrix_system(y_preds: list, y_true: list):
    gender_matrix = confusion_matrix(list(map(lambda x : x['gender'], y_preds)), list(map(lambda x : x['gender'], y_true)), labels=['female', 'male'])
    type_clothes_matrix = confusion_matrix(list(map(lambda x : x['top'][0], y_preds)) + list(map(lambda x : x['bottom'][0], y_preds)),
                                           list(map(lambda x : x['top'][0], y_true)) + list(map(lambda x : x['bottom'][0], y_true)),
                                           labels=['blazer', 'bomber_jacket', 'trousers', 'short', 'skirt', 'sweater', 'tank', 'tee'])
    plot_confusion_matrix(gender_matrix, ['female', 'male'], '/content/gender_matrix.png')
    plot_confusion_matrix(type_clothes_matrix, ['blazer', 'bomber_jacket', 'trousers', 'short', 'skirt', 'sweater', 'tank', 'tee'], '/content/type_clothes_matrix.png')

def plot_confusion_matrix(matrix, labels, save_file):
    sns.set(color_codes=True)
    plt.figure(1, figsize=(15, 10))
 
    plt.title("Confusion Matrix")
 
    sns.set(font_scale=1.4)
    ax = sns.heatmap(matrix, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'}, fmt='')
 
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels, rotation=0)
 
    ax.set(xlabel="True Label", ylabel="Predicted Label")
 
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    plt.close()


# Test
if __name__ == '__main__':
    print(split_label('/content/gdrive/MyDrive/Data_test/All_v3/female--Cardigan-green--Skirt-black--496.jpg'))