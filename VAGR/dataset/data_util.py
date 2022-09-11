import json
import argparse
import json
import os
import re
import sys
from string import punctuation
from xml.dom.minidom import parse

from allennlp.predictors.predictor import Predictor
from nltk.tokenize import TreebankWordTokenizer
from tqdm import tqdm
from lxml import etree

MODELS_DIR = ''
model_path = os.path.join(
      MODELS_DIR, "yelp/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")

def LabeledDataProcess():
      i = 0
      datas = []
      with open("./Laptops_corenlp/Laptop_train.json") as jsonfile:
            data = json.load(jsonfile)
            for line in data:
                  if i > 228:
                        break
                  datas.append(line)
                  i += 1
      with open("./Laptops_corenlp/Laptop_10train.json", "w") as f:
             json.dump(datas, fp=f, indent=6)
      f.close()

def originalDataProcess():
      i = 1
      file = open("../Amazon/Electronics_5.json",'r')
      with open("../Amazon/Electronics_10000.txt", "a") as ftxt:
            for line in file.readlines():
                  if i > 10000:
                    break
                  if i >0:
                        text = json.loads(line)['summary'].replace('\n', ' ')
                        if 5 <= text.split().__len__() <= 100:
                              ftxt.write(text+'\n')
                              i += 1
      ftxt.close()


def UnlabelDataProcess():
      i = 0
      file = open("../Amazon/Electronics_5.json",'r')
      dict = []
      for line in file.readlines():
            i += 1
            if i > 11000:
                  break
            if i > 10000:
                  data = json.loads(line)
                  data.pop("review_id")
                  data.pop("date")
                  data.pop("user_id")
                  data.pop("business_id")
                  data.pop("stars")
                  data.pop("useful")
                  data.pop("funny")
                  data.pop("cool")
                  dicts = {i: '' for i in punctuation}
                  punc_table = str.maketrans(dicts)
                  data["text_list"]  = data["text"].translate(punc_table).split()
                  data["text_list"] = data["text_list"][:128]
                  data["text"] = " ".join(data["text_list"])
                  lens = len(data["text_list"])
                  data["aspect_post"] = [int(lens/2),int(lens/2)+1]
                  dict.append(data)
                  print(lens)
      with open("Review_short1000.json", "w") as f:
             json.dump(dict, fp=f, indent=6)
      f.close()


def UnlabelDataProcess2():
      datas= []
      file = open("tip10000.txt",'r')
      for line in file.readlines():
            data = {}
            data["text"] = line.rstrip()
            data["text_list"] = line.split()
            lens = data["text_list"].__len__()
            data["aspect_post"] = [int(lens / 2), int(lens / 2) + 1]
            datas.append(data)
      with open("Tips_10000.json", "w") as f:
             json.dump(datas, fp=f, indent=6)
      f.close()


def UnlabelDataProcess_Amazon():
      datas= []
      file = open("../Amazon/Electronics_10000.txt",'r')
      for line in file.readlines():
            data = {}
            data["text"] = line.rstrip()
            data["text_list"] = line.split()
            lens = data["text_list"].__len__()
            data["aspect_post"] = [int(lens / 2), int(lens / 2) + 1]
            datas.append(data)
      with open("../Amazon/Electronics_10000.json", "w") as f:
             json.dump(datas, fp=f, indent=6)
      f.close()


def RestaurantDataProcess():
      datas = []
      dom = parse("semeval16/EN_REST_SB1_TEST.xml.gold")
      data = dom.documentElement
      reviews = data.getElementsByTagName('Review')
      for review in tqdm(reviews, total=len(reviews), desc="Data Process"):
            sentences = review.childNodes[1].getElementsByTagName('sentence')
            for sentence in sentences:
                  d = {}
                  pun = punctuation
                  punc_table = str.maketrans(pun, " " * len(pun))
                  d["text"] = sentence.getElementsByTagName('text')[0].childNodes[0].nodeValue.replace("\n"," ").replace("–", " ")
                  d["text_list"] = d["text"].translate(punc_table).split()
                  opinions = sentence.getElementsByTagName('Opinion')
                  for opinion in opinions:
                        d["label"] = opinion.getAttribute('polarity')
                        d["aspect"] = opinion.getAttribute('target').translate(punc_table)
                        if d["aspect"] == 'NULL':
                              break
                        d["aspect_post"] = [d["text_list"].index(d["aspect"].split()[0]),
                                           d["text_list"].index(d["aspect"].split()[-1])+1]

                  datas.append(d)
      with open("../semeval16/R16TestGold.json", "w") as f:
            json.dump(datas, fp=f, indent=6)
      f.close()


def parse_args():
      parser = argparse.ArgumentParser()

      # Required parameters
      parser.add_argument('--model_path', type=str, default=model_path,
                          help='Path to biaffine dependency parser.')
      parser.add_argument('--data_path', type=str, default='review_short10.txt',
                          help='Directory of where semeval14 or twiiter data held.')
      return parser.parse_args()


def text2docs(file_path, predictor):
      '''
      Annotate the sentences from extracted txt file using AllenNLP's predictor.
      '''
      with open(file_path, 'r') as f:
            sentences = f.readlines()
      docs = []
      print('Predicting dependency information...')
      for i in tqdm(range(len(sentences))):
            docs.append(predictor.predict(sentence=sentences[i]))

      return docs


def dependencies2format(doc):  # doc.sentences[i]
      '''
      Format annotation: sentence of keys
                                  - tokens
                                  - tags
                                  - predicted_dependencies
                                  - predicted_heads
                                  - dependencies
      '''
      sentence = {}
      sentence['tokens'] = doc['words']
      sentence['tags'] = doc['pos']
      # sentence['energy'] = doc['energy']
      predicted_dependencies = doc['predicted_dependencies']
      predicted_heads = doc['predicted_heads']
      sentence['predicted_dependencies'] = doc['predicted_dependencies']
      sentence['predicted_heads'] = doc['predicted_heads']
      sentence['dependencies'] = []
      for idx, item in enumerate(predicted_dependencies):
            dep_tag = item
            frm = predicted_heads[idx]
            to = idx + 1
            sentence['dependencies'].append([dep_tag, frm, to])

      return sentence


def get_dependencies(file_path, predictor):
      docs = text2docs(file_path, predictor)
      sentences = [dependencies2format(doc) for doc in docs]
      return sentences


def syntaxInfo2json(sentences, origin_file):
      json_data = []
      tk = TreebankWordTokenizer()
      mismatch_counter = 0
      idx = 0
      with open(origin_file, 'rb') as fopen:
            raw = fopen.read()
            root = etree.fromstring(raw)
            for sentence in root:
                  example = dict()
                  example["sentence"] = sentence.find('text').text

                  # for RAN
                  terms = sentence.find('aspectTerms')
                  if terms is None:
                        continue

                  example['tokens'] = sentences[idx]['tokens']
                  example['tags'] = sentences[idx]['tags']
                  example['predicted_dependencies'] = sentences[idx]['predicted_dependencies']
                  example['predicted_heads'] = sentences[idx]['predicted_heads']
                  example['dependencies'] = sentences[idx]['dependencies']
                  # example['energy'] = sentences[idx]['energy']

                  example["aspect_sentiment"] = []
                  example['from_to'] = []  # left and right offset of the target word

                  for c in terms:
                        if c.attrib['polarity'] == 'conflict':
                              continue
                        target = c.attrib['term']
                        example["aspect_sentiment"].append((target, c.attrib['polarity']))

                        # index in strings, we want index in tokens
                        left_index = int(c.attrib['from'])
                        right_index = int(c.attrib['to'])

                        left_word_offset = len(tk.tokenize(example['sentence'][:left_index]))
                        to_word_offset = len(tk.tokenize(example['sentence'][:right_index]))

                        example['from_to'].append((left_word_offset, to_word_offset))
                  if len(example['aspect_sentiment']) == 0:
                        idx += 1
                        continue
                  json_data.append(example)
                  idx += 1
      extended_filename = origin_file.replace('.xml', '_biaffine_depparsed.json')
      with open(extended_filename, 'w') as f:
            json.dump(json_data, f)
      print('done', len(json_data))
      print(idx)


def main():
      args = parse_args()

      predictor = Predictor.from_path(args.model_path)

      data = [('tip.xml')]
      for train_file in data:

            # txt -> json
            train_sentences = get_dependencies(args.data_path, predictor)
            # test_sentences = get_dependencies(os.path.join(
            #       args.data_path, test_file.replace('.xml', '_text.txt')), predictor)
            dict = []
            print(len(train_sentences))
            for item in train_sentences:
                  item["token"] = item.pop("tokens")
                  item["pos"] = item.pop("tags")
                  item["head"] = item.pop("predicted_heads")
                  item["deprel"] = item.pop("predicted_dependencies")
                  item.pop("dependencies")
                  item['aspects'] = [{"term": [], "from":-1, "to":-1, "polarity":"positive"}]
                  dict.append(item)
            # syntaxInfo2json(train_sentences, args.data_path)
            # syntaxInfo2json(test_sentences, os.path.join(args.data_path, test_file))
      with open("yelp/Review_short10.json", "w") as f:
            json.dump(dict, fp=f, indent=6)


if __name__ == "__main__":
      LabeledDataProcess()