from html import entities
import re
from nltk import tokenize
from bs4 import BeautifulSoup
import os
from nltk.tokenize import word_tokenize
import inflect
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import csv
import shutil
from sklearn.utils import shuffle

def creat_REdata(filename, label):
    with open(filename, 'r') as f:
        data = f.read()
    bs_data = BeautifulSoup(data, 'xml')
    passages = bs_data.find_all('passage')
    offset = 0
    all_text = ""
    all_annotations = []
    refids = []
 
    for passage in passages:
        text = passage.find('text').text
        all_text += text
        anno = passage.find_all('annotation')
        relation = passage.find_all('relation')

        for rel in relation:
            re_type = rel.find('infon', {'key':'type'})
            if re_type.text == str(label):         # 1 is positive relation, 0 is negative
                nodes = rel.find_all('node')
                for node in nodes:
                    refid = int(node['refid'])
                    refids.append(refid)

        offset += int(passage.find('offset').text)
        for an in anno:
            an_type = an.find('infon', {'key':'type'})
            if an_type.text == 'Species' or an_type.text == 'Gene':
                anno_id = int(an['id'])
                location = an.find('location')
                o = int(location['offset'])
                l = int(location['length'])
                t = an.find('text').text

                all_annotations.append((anno_id, o, l, t, an_type.text))

        all_text += " "
    # print('all_anno:', all_annotations)
    all_pair = []
    for i in range(0, len(refids), 2):
        pair_info = []    
        for anno_info in all_annotations:
            anno_id, anno_o, anno_l, anno_word, anno_type = anno_info
            if refids[i] == anno_id:
                pair_info.append(anno_info)
            if refids[i+1] == anno_id:
                pair_info.append(anno_info)
        all_pair.append(pair_info)
    
    sentences_list = tokenize.sent_tokenize(all_text)
    re_text = all_text

    count = 0
    re_sent = []
    for (first_en, second_en) in all_pair:
        first_id, first_o, first_l, first_word, first_type = first_en
        second_id, second_o, second_l, second_word, second_type = second_en
        text_ = re_text[:first_o] + '@' + first_type + '$' + re_text[first_o+first_l:second_o] + '@' + second_type + '$' + re_text[second_o+second_l:]
        all_sent = tokenize.sent_tokenize(text_)
        for sent in all_sent:
            if '@Gene$' in sent and '@Species$' in sent:
                re_sent.append(sent)
    return re_sent, [label]*len(re_sent)

def split_fold(fold_num, save_path):
    df_n = pd.read_csv('./all_negative.tsv', sep='\t')
    df_p = pd.read_csv('./all_positive.tsv', sep='\t')
    all_data = []
    all_idx = [i for i in range(len(df_n))]
    all_p_idx = [i for i in range(len(df_p))]
    kf = KFold(n_splits=fold_num, shuffle=True)

    train_ids, test_ids = [], []
    for train_id, test_id in kf.split(all_idx):
        train_ids.append(train_id)
        test_ids.append(test_id)

    p_train_ids, p_test_ids = [], []
    for train_id, test_id in kf.split(all_p_idx):
        p_train_ids.append(train_id)
        p_test_ids.append(test_id)

    for i in range(fold_num):
        print("fold:", i)
        # print('negative_test_id:', test_ids[i])
        # print('positive_test_id', p_test_ids[i])
        neg_data = df_n.iloc[test_ids[i]] 
        pos_data = df_p.iloc[p_test_ids[i]]
        fold_data = pd.concat([neg_data, pos_data], axis=0, ignore_index=True)
        df = pd.DataFrame(fold_data)
        df.to_csv(save_path + '/fold_' + str(i) + '.tsv', header=['file_name', 'sentence', 'label'], index=False, sep='\t', encoding='utf-8')
    
def create_dataset(fold_path, fold_num, test_fold, val_fold, train_fold_list):
    save_path_dir = os.path.join(fold_path, str(test_fold))
    if not os.path.exists(save_path_dir):
        os.mkdir(save_path_dir)
    val_data = pd.read_csv(os.path.join(fold_path, "fold_"+str(val_fold)+".tsv"), sep='\t', usecols = ['sentence', 'label'])
    val_data = val_data[['sentence','label']]
    val_data.to_csv(os.path.join(save_path_dir, "dev.tsv"), index=False, header=['sentence', 'label'], sep='\t', encoding='utf-8')

    test_data = pd.read_csv(os.path.join(fold_path, "fold_"+str(test_fold)+".tsv"), sep='\t')
    idx = np.array([i for i in range(len(test_data))])
    test_sentences = test_data['sentence']
    f = np.stack((idx, test_sentences), 1)
    log = pd.DataFrame(data = f)
    log.to_csv(os.path.join(save_path_dir, "test.tsv"), index=False, header=['index','sentence'], sep='\t', encoding='utf-8')

    test_labels = test_data['label']
    f1 = np.stack((idx,test_sentences,test_labels), 1)
    log1 = pd.DataFrame(data = f1)
    log1.to_csv(os.path.join(save_path_dir, "test_original.tsv"), index=False, header=['index','sentence','label'], sep='\t', encoding='utf-8')


    train_data = []
    merged = pd.DataFrame()
    for i in train_fold_list:
        dataset = fold_path + "/fold_" + str(i) + ".tsv"
        read_csv = pd.read_csv(dataset, usecols = ['sentence', 'label'], sep='\t')
        train_data.append(read_csv)
    csv_merge = pd.concat(train_data, axis=0)
    csv_merge = shuffle(csv_merge)
    csv_merge.to_csv(os.path.join(save_path_dir, "train.tsv"), header=None, sep='\t', index=False, encoding='utf-8')


def count_labels(data_path, fold_num):
    count_zero, count_one = 0, 0
    for i in range(1, fold_num+1):
        path = os.path.join(data_path, str(i), 'train_original.tsv')
        print(path)
        data = pd.read_csv(path, sep='\t')
        count_labels = data['label'].value_counts(ascending=True)
        count_zero += count_labels[0]
        count_one += count_labels[1]
    print('number of label 0 : ', count_zero)
    print('number of label 1 : ', count_one)
        
def creat_LOOCV(save_path):
    import csv
    all_data = []
    data = pd.read_csv('./all_data.tsv', sep='\t')
    for i in range(len(data)):
        print(i)
        save_path_dir = os.path.join(save_path, str(i))
        if not os.path.exists(save_path_dir):
            os.mkdir(save_path_dir)
        test = [data['sentence'].iloc[i]]        
        idx = np.array([0])
        
        f = np.stack((idx, test), 1)
        log = pd.DataFrame(data = f)
        log.to_csv(os.path.join(save_path_dir, "test.tsv"), index=False, header=['index','sentence'], sep='\t', encoding='utf-8')
        
        test_l = [data['label'].iloc[i]]
        f1 = np.stack((idx,test,test_l), 1)
        log1 = pd.DataFrame(data = f1)
        log1.to_csv(os.path.join(save_path_dir, "test_original.tsv"), index=False, header=['index','sentence','label'], sep='\t', encoding='utf-8')
        log1.to_csv(os.path.join(save_path_dir, "dev.tsv"), index=False, header=['index','sentence','label'], sep='\t', encoding='utf-8')


        train = pd.concat([data.iloc[0:i],data.iloc[i+1:]], axis=0)
        train = shuffle(train)
        train = train[['sentence', 'label']]
        train.to_csv(os.path.join(save_path_dir, "train.tsv"), header=None, sep='\t', index=False, encoding='utf-8')

      



if __name__ == '__main__':
    ## preprocess TeamTat data
    all_neg_xml, all_neg_sents, all_neg_labels = [], [], []
    all_pos_xml, all_pos_sents, all_pos_labels = [], [], []
    for i in range(10):
        files = os.listdir('./RE_data/fold'+str(i))
        for file in files:
            print(file)
            # negative
            neg_sents, neg_labels = creat_REdata(os.path.join('./RE_data/fold'+str(i), file), 0) 
            all_neg_sents += neg_sents
            all_neg_labels += neg_labels
            all_neg_xml += len(neg_sents)*[file]
            # positive
            pos_sents, pos_labels = creat_REdata(os.path.join('./RE_data/fold'+str(i), file), 1) 
            all_pos_sents += pos_sents
            all_pos_labels += pos_labels
            all_pos_xml += len(pos_sents)*[file]
            
    neg_xml_list = np.array(all_neg_xml)
    neg_sentence_list = np.array(all_neg_sents)
    neg_label_list = np.array(all_neg_labels)
    name = ['file_name', 'sentence', 'label']
    f = np.stack((neg_xml_list, neg_sentence_list, neg_label_list), 1)
    log = pd.DataFrame(data = f)
    log.to_csv('./all_negative.tsv', index=False, header=name, sep='\t', encoding='utf-8')

    pos_xml_list = np.array(all_pos_xml)
    pos_sentence_list = np.array(all_pos_sents)
    pos_label_list = np.array(all_pos_labels)
    fp = np.stack((pos_xml_list, pos_sentence_list, pos_label_list), 1)
    log_p = pd.DataFrame(data = fp)
    log_p.to_csv('./all_positive.tsv', index=False, header=name, sep='\t', encoding='utf-8')

    all_xml_list = np.array(all_pos_xml+all_neg_xml)
    all_sentence_list = np.array(all_pos_sents+all_neg_sents)
    all_label_list = np.array(all_pos_labels+all_neg_labels)
    fa = np.stack((all_xml_list, all_sentence_list, all_label_list), 1)
    log_a = pd.DataFrame(data = fa)
    log_a.to_csv('./all_data.tsv', index=False, header=name, sep='\t', encoding='utf-8')

    ## split 5 folds/ 10 folds
    split_fold(5, './dataset')
    ## creat datasets
    test_list = [0,1,2,3,4]
    val_list = [1,2,3,4,0]
    # test_list = [0,1,2,3,4,5,6,7,8,9]
    # val_list = [1,2,3,4,5,6,7,8,9,0]
    for i in range(len(test_list)):
        val_num = val_list[i]
        test_num = test_list[i]
        train_num = [i for i in range(len(test_list))]
        train_num.remove(val_num)
        train_num.remove(test_num)
        print('test_num: ',test_num, 'val_num: ',val_num ,'train_num: ', train_num)
        create_dataset('./dataset', 5, test_num, val_num, train_num)
    ## create LOOCV dataset
    creat_LOOCV('./dataset/LOOCV')