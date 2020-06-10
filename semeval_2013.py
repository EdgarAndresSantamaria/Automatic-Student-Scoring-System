# XML treatment imports
from os import listdir,path
import xml.etree.ElementTree as ET
# Deep learning imports
import json, random, collections, re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
# chunking imports
from flair.data import Sentence
from flair.models import SequenceTagger
from segtok.segmenter import split_single
# similarity imports (estimator)
from semantic_text_similarity.models import WebBertSimilarity
# retrieve / save models (de)serializtion protocol
import pickle
import pprint
# feedback
import nltk
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
from itertools import combinations, permutations


'''
Paths: Here we define the paths used by the whole script
'''
# this is the only line should be changed for the path indexing (maintaining the project proposed structure)
general_path = "/media/edgar/407d4115-9ff4-45c6-9279-01b62aee0730/similarity_deep_model-master/"
data_path = general_path + "data/"
data_preprocessed_path = general_path + "intermedium/"
embeddings_path = general_path + "embeddings/glove.6B.50d.txt"
output_figures_path = general_path + "output/"
model_path = general_path + "model/"

class Preprocessor:
    '''

    Preprocessor class: in this class is treated every information required from SemEval2013 corpus
    we aim to adequate it for Deep Learning purposes, also provides loading functions for the Deep
    Learning required resources (word embeddings, JSON, padding ...)

    '''
    def __init__(self, sequence_length, preprocess, labelmap):
        self.labelmap = labelmap
        self.preprocess = preprocess
        if  self.preprocess:
            self.posTagger = SequenceTagger.load('pos')
            self.chunker = SequenceTagger.load('chunk')

        # those properties may be also helpful but expensive to calculate
        # self.semTagger = SequenceTagger.load('frame')
        # self.nerTagger = SequenceTagger.load('ner')
        # self.polarityClassifier = TextClassifier.load('en-sentiment')
        self.Sentence_len = sequence_length

    # semeval experiment

    def load_embeddings(self, glove_home, words_to_load = 400000):
        with open(glove_home) as f:
            loaded_embeddings = np.zeros((len(word_indices), 50), dtype='float32')
            for i, line in enumerate(f):
                if i >= words_to_load:
                    break
                s = line.split()
                if s[0] in word_indices:
                    loaded_embeddings[word_indices[s[0]], :] = np.asarray(s[1:])
        return loaded_embeddings

    def sentences_to_padded_index_sequences(self, datasets):
        sntLen = []
        '''Annotates datasets with feature vectors.'''
        PADDING = "<PAD>"
        UNKNOWN = "<UNK>"
        # Extract vocabulary
        def tokenize(string):
            string = re.sub(r'\(|\)', '', string)
            return string.lower().split()

        word_counter = collections.Counter()
        for example in datasets[0]:
            word_counter.update(tokenize(example['reference_answer']))
            word_counter.update(tokenize(example['student_answer']))

        vocabulary = set([word for word in word_counter])
        vocabulary = list(vocabulary)
        vocabulary = [PADDING, UNKNOWN] + vocabulary

        word_indices = dict(zip(vocabulary, range(len(vocabulary))))
        indices_to_words = {v: k for k, v in word_indices.items()}

        for i, dataset in enumerate(datasets):
            for example in dataset:
                for sentence in ['reference_answer', 'student_answer']:
                    example[sentence + '_index_sequence'] = np.zeros(( self.Sentence_len), dtype=np.int32)

                    token_sequence = tokenize(example[sentence])
                    padding =  self.Sentence_len - len(token_sequence)
                    sntLen . append(len(token_sequence))

                    for i in range( self.Sentence_len):
                        if i >= padding:
                            if token_sequence[i - padding] in word_indices:
                                index = word_indices[token_sequence[i - padding]]
                            else:
                                index = word_indices[UNKNOWN]
                        else:
                            index = word_indices[PADDING]
                        example[sentence + '_index_sequence'][i] = index

        print("The larger sentence has {} tokens".format(max(sntLen)))
        return indices_to_words, word_indices

    def load_semeval(self, snli_home):
        LABEL_MAP = self.labelmap

        def load_snli_data(path):
            data = []
            with open(path) as f:
                for line in f:
                    loaded_example = json.loads(line)
                    if loaded_example["grade"] not in LABEL_MAP:
                        print(loaded_example["grade"]) # checking 4 not mapped labels
                    loaded_example["label"] = LABEL_MAP[loaded_example["grade"]]
                    data.append(loaded_example)
                random.seed(1)
                random.shuffle(data)
            return data

        return load_snli_data(snli_home + '/train.jsonl'), load_snli_data(snli_home + '/dev.jsonl'), load_snli_data(snli_home + '/test.jsonl')

    def XML_2_JSON(self, bettle, path):
        jsonl = []
        cnt = 0
        for file in listdir(path):
            if (file.endswith(".xml")):
                cnt += 1
                intern_path = path + file
                #print(intern_path)
                try:
                    tree = ET.parse(intern_path)
                    root = tree.getroot()
                    question = root.find("questionText").text
                    if bettle:  # match answers
                        ref_answers = root.findall('referenceAnswers/referenceAnswer')
                        for ref in ref_answers:
                            id2_search = ref.attrib['id']
                            # category = ref.attrib['category']
                            stu_answers = root.findall(
                                "./studentAnswers/studentAnswer[@answerMatch='{}']".format(id2_search))
                            for stu in stu_answers:
                                status = stu.attrib['accuracy']
                                # print(" \n \n {} \n {} \t\t {} \n {} \t\t {} \t\t {}".format(question, ref.text, stu.text, status, id2_search, category))
                                if self.preprocess:
                                    preporcesed_ref = self.getChunks(ref.text)
                                    preporcesed_answer = self.getChunks(stu.text)
                                    json_set = {"reference_answer": ref.text, "reference_answer_pre":preporcesed_ref[0].to_tagged_string(),"student_answer_pre":preporcesed_answer[0].to_tagged_string(), "student_answer": stu.text, "grade": status, "question": question}
                                else:
                                    json_set = {"reference_answer": ref.text,
                                                "student_answer": stu.text, "grade": status, "question": question}

                                jsonl.append(json_set)

                    silly_answers = root.findall("./studentAnswers/studentAnswer")
                    for answer in silly_answers:
                        try:
                            answer.attrib['answerMatch']
                            # already treated aligned answer
                            pass
                        except:  # sciEntsBank and Bettle answers without alignment
                            status = answer.attrib['accuracy']
                            ref_answers = root.findall('referenceAnswers/referenceAnswer')
                            for ref in ref_answers:  # align as incorrect to all golden (to clarify that are not aligned
                                if self.preprocess:
                                    preporcesed_ref = self.getChunks(ref.text)
                                    preporcesed_answer = self.getChunks(answer.text)
                                    json_set = {"reference_answer_pre":preporcesed_ref[0].to_tagged_string(),"student_answer_pre":preporcesed_answer[0].to_tagged_string(), "reference_answer": ref.text, "student_answer": answer.text, "grade": status, "question": question}
                                else:
                                    json_set = {"reference_answer": ref.text, "student_answer": answer.text,
                                                "grade": status, "question": question}

                                jsonl.append(json_set)
                except:
                    print("problems in " + file)
        return jsonl

    def getChunks(self, text):
        tagged_txt = list()
        for part in split_single(text):
            sentence = Sentence(str(part))
            self.posTagger.predict(sentence)
            self.chunker.predict(sentence)

            # those properties may be also helpful but expensive to calculate
            #self.nerTagger.predict(sentence)
            #self.semTagger.predict(sentence)
            #self.polarityClassifier.predict(sentence)

            tagged_txt.append(sentence)
        return tagged_txt

    def split_semeval(self, input_path, output_path):
        # Bettle  .... sciEntsBank  loader
        jsonl = []
        jsonl += self.XML_2_JSON(False, input_path+"sciEntsBank/")
        jsonl += self.XML_2_JSON(True, input_path+"beetle/")
        #split data
        train_split =int(len(jsonl)*0.6)
        dev_test_split = int(len(jsonl)*0.2)
        jsonl_train = jsonl[:train_split]
        jsonl_dev = jsonl[train_split:train_split + dev_test_split]
        jsonl_test = jsonl[train_split+ dev_test_split:]
        #generate output json
        with open(output_path+'train.jsonl', 'w') as f:
            for cur_json in jsonl_train:
                f.write(json.dumps(cur_json)+ "\n")
        with open(output_path+'dev.jsonl', 'w') as f:
            for cur_json in jsonl_dev:
                f.write(json.dumps(cur_json) + "\n")
        with open(output_path+'test.jsonl', 'w') as f:
            for cur_json in jsonl_test:
                f.write(json.dumps(cur_json) + "\n")

# shared vars
Sentence_len = 20 # larger sentence 110
# ----------------------------------------------------------------
# Preprocess data (calculates chunking and POS tagging with FLAIR so may take it's time)
#preprocessor = Preprocessor(Sentence_len, preprocess=True)
# generate preprocessed data
#preprocessor.split_semeval(input_path="/home/edgar/Escritorio/semeval-2013-2-and-3-way/training/2way/",output_path="/home/edgar/Escritorio/semeval-2013-2-and-3-way/")

# ----------------------------------------------------------------
# Lauch loaders
label2ind = {"correct": 0, "incorrect": 1}
ind2label = {0: "correct", 1: "incorrect"}
preprocessor = Preprocessor(Sentence_len, preprocess=False, labelmap=label2ind)
preprocessor.split_semeval(input_path=data_path, output_path=data_preprocessed_path)
# load the data
training_set, dev_set, test_set = preprocessor.load_semeval(snli_home=data_preprocessed_path)
# adequate data for the task
indices_to_words, word_indices = preprocessor.sentences_to_padded_index_sequences([training_set, dev_set, test_set])
# load embeddings for the task
loaded_embeddings = preprocessor.load_embeddings(glove_home=embeddings_path)

class SimmilarityClassifier():

    def __init__(self, vocab_size, sequence_length, labelmap, test = False):
        self.labelmap = labelmap
        self.web_model = WebBertSimilarity(device='cpu', batch_size=1000)  # defaults to GPU prediction
        # Define the hyperparameters
        if test:
            self.training_epochs = 1  # How long to train for - chosen to fit within class time
        else:
            self.training_epochs = 100  # How long to train for - chosen to fit within class time
        self.display_epoch_freq = 1  # How often to print cost (in epochs)
        self.display_step_freq = 11  # How often to test (in steps)
        self.dim = sequence_length  # The dimension of the hidden state of each RNN
        self.embedding_dim = 50  # The dimension of the learned word embeddings
        self.batch_size = 16  # Somewhat arbitrary - can be tuned, but often tune for speed, not accuracy
        self.vocab_size = vocab_size  # Defined by the file reader above
        self.sequence_length = sequence_length  # Defined by the file reader above
        self.step = 1
        self.epoch = 0
        self.max_patience = 5
        self.l2_lambda = 0.001
        # Define the parameters
        self.trainable_variables = []
        self.E = tf.Variable(loaded_embeddings, trainable=False)
        self.trainable_variables.append(self.E)
        # Define the parameters of the GRUs
        #  - Note that we need to learn two GRUs:
        #  - Use Figure above to understand how you can organize you GRU models.
        # params for encoding (premise) , decoding (hypothesis) and attention Weights (attention) GRU
        self.W_rnn = {}
        self.W_r = {}
        self.W_z = {}
        self.b_rnn = {}
        self.b_r = {}
        self.b_z = {}

        # - You can re-use part of you code in lab5
        for self.name in ['p', 'h', 'm']:
            # but you need to set different dimensionality to each GRU...
            if self.name == 'm':
                # 2) for maching GRU. Hint: What is the dimensionality of input??
                # input 4m the atention weigths??
                in_dim = 2 * self.dim
                out_dim = 0
                # in_dim = 2*self.dim
            elif self.name == 'p':
                # 2) for maching GRU. Hint: What is the dimensionality of input??
                # input 4m the atention weigths??
                out_dim = self.dim
                in_dim = self.embedding_dim + self.dim
                # in_dim = 2*self.dim
            else:
                # 1) for encoding premise and hypothesis
                in_dim = self.embedding_dim
                out_dim = 0
            # init GRU params ... check lab 5

            self.W_rnn[self.name] = tf.Variable(tf.random.normal([in_dim + self.dim, out_dim + self.dim], stddev=0.1))
            self.b_rnn[self.name] = tf.Variable(tf.random.normal([self.dim + out_dim], stddev=0.1))
            self.trainable_variables.append(self.W_rnn[self.name])
            self.trainable_variables.append(self.b_rnn[self.name])

            self.W_r[self.name] = tf.Variable(tf.random.normal([in_dim + self.dim, out_dim + self.dim], stddev=0.1))
            self.b_r[self.name] = tf.Variable(tf.random.normal([self.dim + out_dim], stddev=0.1))
            self.trainable_variables.append(self.W_r[self.name])
            self.trainable_variables.append(self.b_r[self.name])

            self.W_z[self.name] = tf.Variable(tf.random.normal([in_dim + self.dim, out_dim + self.dim], stddev=0.1))
            self.b_z[self.name] = tf.Variable(tf.random.normal([self.dim + out_dim], stddev=0.1))
            self.trainable_variables.append(self.W_z[self.name])
            self.trainable_variables.append(self.b_z[self.name])

        # Define the attention parameters.
        #  - Attention parameters: You need to define just one variable that learns matching
        #    premise and hypothesis sequences.
        self.attn = tf.Variable(tf.random.normal([2 * self.dim, 2 * self.dim], stddev=0.1))
        self.trainable_variables.append(self.attn)

        #  - Attention score is defined in eq. 8 in Luong et al. (general score)
        #  - This simplify eq.6 in Wang and Jiang for the attention score.
        #    self.score = tf.Variable(tf.random.normal([self.dim , self.dim], stddev=0.1))
        #  - Hint: What are the dimensionality of vectors of both side in the equation?

        # Define the paremeters for the classification layer (as in Lab5).
        self.w_cl = tf.Variable(tf.random.normal([self.dim, 2], stddev=0.1))
        self.trainable_variables.append(self.w_cl)
        self.b_cl = tf.Variable(tf.random.normal([2], stddev=0.1))
        self.trainable_variables.append(self.b_cl)

    def package_params(self):
        return self.trainable_variables

    def setup_params(self, trainable_variables):
        ind = 0
        self.trainable_variables = trainable_variables
        self.E = trainable_variables[ind]
        ind +=1
        for self.name in ['p', 'h', 'm']:
            self.W_rnn[self.name] = trainable_variables[ind]
            ind += 1
            self.b_rnn[self.name] = trainable_variables[ind]
            ind += 1
            self.W_r[self.name] = trainable_variables[ind]
            ind += 1
            self.b_r[self.name] = trainable_variables[ind]
            ind += 1
            self.W_z[self.name] = trainable_variables[ind]
            ind += 1
            self.b_z[self.name] = trainable_variables[ind]
            ind += 1

        self.attn = trainable_variables[ind]
        ind += 1
        self.w_cl = trainable_variables[ind]
        ind += 1
        self.b_cl = trainable_variables[ind]

    def generate_assessment(self, examples, autocorrelation):
        for n, example in enumerate(examples):
            reference_ans = example["reference_answer"]
            student_ans =  word_tokenize(example["student_answer"])
            importance_vec = tf.unstack(tf.reduce_mean(autocorrelation[n], 0))
            AnsWords = []
            factors = [] # mask to detect the words to highlight on red (2) and green (1)
            # discriminate wrong / correct words by correlation
            # extract answer keywords
            for ind, weight in enumerate(importance_vec): # retrieve incorrect words
                if weight < 0:
                    factors.append(2)
                    AnsWords.append(" ")
                elif  weight > 0:
                    factors.append(1)
                    AnsWords.append(student_ans[ind])

            if example['estimated_score'] == 'incorrect':
                # retrieve all possible correct senses for the reference (track reference variability)
                list_lemmas = [reference_ans]
                tokenized_sent = word_tokenize(reference_ans)
                for i, token in enumerate(tokenized_sent):  # desambiguate semantic meaning using LESK (could be improved)
                    synset = lesk(tokenized_sent, token)
                    if synset:
                        possible_lemmas = [" ".join(tokenized_sent[:i] + [lemma.name().lower()] + tokenized_sent[i + 1:]) for lemma in
                                           synset.lemmas()]
                        for lemma in possible_lemmas:
                            similarity = self.web_model.predict([(reference_ans, lemma)])  # similarity range [0..5]
                            if (similarity[0] > 4.7) and (lemma != reference_ans) and (lemma not in list_lemmas):  # if sufficient similarity confidence add new example
                                list_lemmas.append(lemma)

                # extract suitable combinations for certain keywords in front of a reference
                def getFeedback(RefWords, AnsWords,reference_ans):
                    RefWords = [word.lower() for word in RefWords if word not in AnsWords]
                    fixed = 5

                    ans2Improve = list(permutations(AnsWords, len(AnsWords)))
                    ans2Improve = [list(tp) for tp in ans2Improve]

                    combi = []
                    for i in range(fixed):
                        tmp_combi = list(combinations(RefWords, i+1))
                        tmp_combi = [list(tp) for tp in tmp_combi]
                        combi = combi + tmp_combi

                    list2check = []
                    for possible in ans2Improve:
                        substitude = possible.index(" ")
                        for comb in combi:
                             tmp = possible.copy()
                             check = tmp[:substitude] + comb + tmp[substitude+1:]
                             list2check. append(check)


                    mapped_comb = list(map(lambda x: " ".join(x), list2check))
                    ref_list = [reference_ans] * len(mapped_comb)
                    zipped_batch = list(zip(mapped_comb, ref_list))
                    results = self.web_model.predict(zipped_batch[:1000])  # check 250 combinations (optimization pourposes)

                    feedback_ideas = []
                    for i, tmp in enumerate(results):
                        tuple = zipped_batch[i]
                        feedback = tuple[0].lower()
                        if (tmp > 4.) and (feedback != reference_ans) and (
                                feedback not in feedback_ideas):  # if sufficient similarity confidence add new example
                                    tuple = (feedback,tmp)
                                    feedback_ideas.append(tuple)
                    return feedback_ideas

                # extract keywords from reference
                RefWords = []
                for lemma in list_lemmas:
                    solution2fit = word_tokenize(lemma)
                    for token in solution2fit:
                        if token not in RefWords:
                            RefWords.append(token)

                punct_marks = ['.','"',')','(',':','-','_',';',"'",'@', '[', ']',',']
                RefWords = [word.lower() for word in RefWords if word not in punct_marks]# remove punctuation marks
                RefWords = list(dict.fromkeys(RefWords))  # avoid repeated words
                AnsWords = [word.lower() for word in AnsWords if word not in punct_marks]# remove punctuation marks
                AnsWords = list(dict.fromkeys(AnsWords))  # avoid repeated words
                feedback_ideas = getFeedback(RefWords, AnsWords,  reference_ans)

                if (not feedback_ideas) :
                    feedback_ideas = reference_ans
                else:
                    def sort_feedback(list_of_tuples):  # sort the list by the
                        list_of_tuples.sort(key=lambda x: x[1], reverse=True)
                        return list_of_tuples

                    feedback_ideas = sort_feedback(feedback_ideas)
                    feedback_ideas = feedback_ideas[0]

            else:
                if example['estimated_mark'] > 8:
                    feedback_ideas = "very good, congratulations !"
                elif example['estimated_mark'] > 6:
                    feedback_ideas = "good !"
                else:
                    feedback_ideas = "passed, check the review"

            tmp = {"feedback": feedback_ideas, "highlight_vector": factors}
            example.update(tmp)
            examples[n] = example

        return examples

    # define the GRU function (Hint: check lab 5)
    # todo update into LSTM
    def gru(self, emb, h_prev, name):
        emb_h_prev = tf.concat([emb, h_prev], 1, name=name + '_emb_h_prev')
        z = tf.nn.sigmoid(tf.matmul(emb_h_prev, self.W_z[name]) + self.b_z[name], name=name + '_z')
        r = tf.nn.sigmoid(tf.matmul(emb_h_prev, self.W_r[name]) + self.b_r[name], name=name + '_r')
        emb_r_h_prev = tf.concat([emb, r * h_prev], 1, name=name + '_emb_r_h_prev')
        h_tilde = tf.nn.tanh(tf.matmul(emb_r_h_prev, self.W_rnn[name]) + self.b_rnn[name], name=name + '_h_tilde')
        h = (1. - z) * h_prev + z * h_tilde
        return h

    def evaluate_classifier(self, eval_set):
        correct = 0
        hypotheses = self.classify(eval_set)
        for i, example in enumerate(eval_set):
            hypothesis = hypotheses[i]
            if hypothesis == example['label']:
                correct += 1
        return correct / float(len(eval_set))

    def estimate_grade(self, examples, output_dir, title):
        attn_weights = self.get_attn(examples)
        avg_correlation_list = []
        avg_weights_list = []
        autocorrelation_list = []
        for i in range(len(examples)):
            # x axis
            premise_tokens = [indices_to_words[index] for index in examples[i]['reference_answer_index_sequence']]
            # y axis
            hypothesis_tokens = [indices_to_words[index] for index in examples[i]['student_answer_index_sequence']]

            def pad_identifier(tokens):
                start_ind = 0
                for item in tokens: # avoid initial padding x_axis
                    if item.__contains__("<PAD>"):
                        start_ind += 1
                    else:
                        break
                return start_ind

            x_start_ind = pad_identifier(premise_tokens)
            y_start_ind = pad_identifier(hypothesis_tokens)

            # Basic Configuration
            fig = plt.figure(figsize=(16, 16))

            # lines. (premise stats)
            deviation_premise = tfp.stats.stddev(attn_weights[i, x_start_ind:, y_start_ind:],1)
            variance_premise = tfp.stats.variance(attn_weights[i, x_start_ind:, y_start_ind:],1)

            ax1 = fig.add_subplot(2,2,3)
            ax1.scatter(premise_tokens[x_start_ind:], deviation_premise, label="deviation")
            ax1.scatter(premise_tokens[x_start_ind:], variance_premise, label="variance")
            plt.setp(ax1.get_xticklabels(), rotation=-45, ha='left', rotation_mode='anchor')
            ax1.legend()

            # lines. (hypothesis stats)
            deviation_hyphotesis= tfp.stats.stddev(attn_weights[i, x_start_ind:, y_start_ind:],0)
            variance_hyphotesis = tfp.stats.variance(attn_weights[i, x_start_ind:, y_start_ind:],0)
            ax4 = fig.add_subplot(2, 2, 4)

            ax4.scatter(hypothesis_tokens[y_start_ind:], deviation_hyphotesis, label="deviation")
            ax4.scatter(hypothesis_tokens[y_start_ind:], variance_hyphotesis, label="variance")
            plt.setp(ax4.get_xticklabels(), rotation=-45, ha='left', rotation_mode='anchor')
            ax4.legend()

            # heat map
            # data to show (pearson correlation)
            ax2 = fig.add_subplot(2, 2, 1)
            pearson_correlation = tfp.stats.auto_correlation(attn_weights[i, x_start_ind:, y_start_ind:])
            autocorrelation_list.append(pearson_correlation)
            im2 = ax2.matshow(pearson_correlation, cmap='coolwarm')

            # Formatting for heat map 1.
            ax2.set_yticks(range(len(premise_tokens[x_start_ind:])))
            ax2.set_xticks(range(len(hypothesis_tokens[y_start_ind:])))
            ax2.set_yticklabels(premise_tokens[x_start_ind:])
            ax2.set_xticklabels(hypothesis_tokens[y_start_ind:])
            ax2.set_title("pearson correlation degree", y=-0.1)
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')
            plt.colorbar(im2, fraction=0.045, pad=0.05, ax=ax2)

            # heat map (used weigths)
            focused_weigths = attn_weights[i, x_start_ind:, y_start_ind:]
            ax3 = fig.add_subplot(2, 2, 2)
            im3 = ax3.matshow(focused_weigths, cmap=plt.cm.inferno)

            # Formatting for heat map 2.
            ax3.set_yticks(range(len(premise_tokens[x_start_ind:])))
            ax3.set_xticks(range(len(hypothesis_tokens[y_start_ind:])))
            ax3.set_yticklabels(premise_tokens[x_start_ind:])
            ax3.set_xticklabels(hypothesis_tokens[y_start_ind:])
            ax3.set_title("weigths plotting", y=-0.1)
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')
            plt.colorbar(im3, fraction=0.045, pad=0.05, ax=ax3)

            fig.tight_layout()
            plt.savefig(output_dir + title + str(i) + '_similarity.pdf')
            plt.clf()
            plt.cla()
            plt.close()

            avg_correlation = tf.reduce_sum(pearson_correlation) / (len(pearson_correlation) * len(pearson_correlation[0]))
            avg_correlation_list.append(avg_correlation)

            avg_weights = tf.reduce_sum(focused_weigths) / (len(focused_weigths) * len(focused_weigths[0]))
            avg_weights_list.append(avg_weights)

        with open(output_dir + title + str(i) + "_output.log", "w") as f:
            f.write("\n\n correlation report: \n")
            f.write(" question: {}".format(examples[0]["question"]))
            for n, example in enumerate(examples):
                similarity = self.web_model.predict([(example["reference_answer"], example["student_answer"])])
                grade = self.classify([example])
                grade = self.labelmap[int(grade)]
                if grade == "incorrect":
                    estimated_mark = int(similarity)
                else:
                    estimated_mark = int(5+similarity)
                tmpDict = {
                    "reference": example["reference_answer"],
                    "answer": example["student_answer"],
                    "score": example["grade"],
                    "estimated_similarity": similarity[0],
                    "estimated_score": grade,
                    "estimated_mark": estimated_mark,
                }
                example.update(tmpDict)
                examples[n] = example

        return examples, autocorrelation_list

    # Define the model: Complete the functions
    # paper = REASONING ABOUT ENTAILMENT WITH NEURAL ATTENTION (Rocktäschel's)
    def model(self, premise_x, hypothesis_x):
        # todo apply dropout technique to avoid overfitting (think about how ...)

        def premise_step(x, h_prev):
            # - Note that attention mechanism is inside premise step (local calculation).
            emb = tf.nn.embedding_lookup(self.E, x)
            h_current = self.gru(emb, h_prev, 'p')
            # following attention based models (light ver.)
            projected_premise = tf.matmul(h_current, self.attn)  # local attention (W * Hs_j)
            return h_current, projected_premise

        def hypothesis_step(x, h_prev, h_attn_k_prev, premise_steps, projected_premise_steps):
            # - Note that attention mechanism is inside hypothesis step (global calculation).
            # the attention is handled every step on hattn_k
            # calculate current layer representation ht_k
            emb = tf.nn.embedding_lookup(self.E, x)
            ht_k = self.gru(emb, h_prev, 'h')
            # paper = Effective Approaches to Attention-based Neural Machine Translation
            # calculate current attention representation h_m (global attention)
            # Equation (6) but differently handled
            h_current_prev_attn = tf.concat([ht_k, h_attn_k_prev], -1)
            # Equation (explained above)
            e_k = h_current_prev_attn * projected_premise_steps
            # Equation (3)
            alfas = tf.nn.softmax(e_k)
            # (premise_max_len, batch_size)
            # Equation (2)
            attn_weights = alfas * premise_steps
            h_attn_k = sum([weight for weight in attn_weights])
            # paper = Learning Natural Language Inference with LSTM (Wang and Jiang)
            # Equation (8)
            h_m = self.gru(h_attn_k, h_attn_k_prev, 'm')
            return h_m, ht_k
            # Split up the inputs into individual tensors

        self.x_premise_slices = tf.split(premise_x, self.sequence_length, 1)
        self.x_hypothesis_slices = tf.split(hypothesis_x, self.sequence_length, 1)
        # Unroll the first RNN (Premise). (Hint: check lab 5)
        self.prem_zero = tf.zeros(tf.stack([tf.shape(premise_x)[0], 2 * self.dim]))
        # Unroll the first RNN
        premise_h_prev = self.prem_zero
        premise_steps = []  # Y input into attention (Hs)
        projected_premise_steps = []
        for t in range(self.sequence_length):
            x_t = tf.reshape(self.x_premise_slices[t], [-1])
            # 1.calculate the state of every premise step.
            # calculate a part of the function of the bilinear attention scoring of each time step: projected_premise_step
            premise_h_prev, projected_premise = premise_step(x_t, premise_h_prev)
            # keep all the premise steps and projected premise steps
            premise_steps.append(premise_h_prev)
            projected_premise_steps.append(projected_premise)
            # Unroll the second RNN (Hypothesis). (Hint: check lab 5)

        self.hyp_zero = tf.zeros(tf.stack([tf.shape(hypothesis_x)[0], self.dim]))
        self.h_prev_attn_weights = tf.zeros(
            tf.stack([tf.shape(hypothesis_x)[0], self.dim]))  # which is the shape of attention word 2 word ??
        # re-init because really h_prev_attn_weights is learning (word2word), we extract the result from there
        h_prev_hypothesis = self.hyp_zero
        h_prev_attn_weights = self.h_prev_attn_weights
        attn_weights_steps = []
        for t in range(self.sequence_length):
            x_t = tf.reshape(self.x_hypothesis_slices[t], [-1])
            # unroll step decoder
            # 1. encode the hypothesis steps with the attention.
            h_prev_attn_weights, h_prev_hypothesis = hypothesis_step(x_t, h_prev_hypothesis, h_prev_attn_weights,
                                                                     premise_steps, projected_premise_steps)
            attn_weights_steps.append(h_prev_attn_weights)
            # 2. keep the attention weights of each step.

        logits = tf.matmul(h_prev_attn_weights, self.w_cl) + self.b_cl  # compute logits for classification (3 classes)
        attn = tf.stack(attn_weights_steps, 1)  # stack weights for plotting functions (example_len, sequence_len, dim)
        # we provide logits as prediction function (over last attn_step) and attn_steps as attention word2word for each premise | hypothesis
        return logits, attn

    def train(self, training_data, dev_data):
        print('Training.')
        self.best_dev = 0
        # Training cycle
        train_acc = []
        dev_acc = []
        epochs = []
        # Training cycle
        for _ in range(self.training_epochs):
            random.shuffle(training_data)
            avg_cost = 0.
            total_batch = int(len(training_data) / self.batch_size)

            # Loop over all batches in epoch
            for i in range(total_batch):
                # Assemble a minibatch of the next B examples
                minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels = self.get_minibatch(
                    training_data, self.batch_size * i, self.batch_size * (i + 1))

                # Run the optimizer to take a gradient step, and also fetch the value of the
                # cost function for logging
                with tf.GradientTape() as tape:
                    logits, _ , = self.model(minibatch_premise_vectors, minibatch_hypothesis_vectors)
                    # implement L2 regularizers (avoid overfitting)
                    l2_cost = self.l2_lambda * (tf.reduce_sum(tf.square(self.W_rnn['p'])) +
                                                tf.reduce_sum(tf.square(self.W_rnn['h'])) +
                                                tf.reduce_sum(tf.square(self.W_rnn['m'])) +
                                                tf.reduce_sum(tf.square(self.W_r['p'])) +
                                                tf.reduce_sum(tf.square(self.W_r['h'])) +
                                                tf.reduce_sum(tf.square(self.W_r['m'])) +
                                                tf.reduce_sum(tf.square(self.W_z['p'])) +
                                                tf.reduce_sum(tf.square(self.W_z['h'])) +
                                                tf.reduce_sum(tf.square(self.W_z['m'])) +
                                                tf.reduce_sum(tf.square(self.w_cl)) +
                                                tf.reduce_sum(tf.square(self.attn))
                                                )
                    # costs curve ...
                    # Define the cost function (here, the softmax exp and sum are built in)
                    total_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=minibatch_labels)+l2_cost)

                # This  performs the main SGD update equation with gradient clipping using Adam algorithm
                optimizer_obj = tf.optimizers.Adam()
                gradients = tape.gradient(total_cost, self.trainable_variables)
                gvs = zip(gradients, self.trainable_variables)
                capped_gvs = [(tf.clip_by_norm(grad, 5.0), var) for grad, var in gvs if grad is not None]
                optimizer_obj.apply_gradients(capped_gvs)
                # averaged costs curve ...
                avg_cost += total_cost / total_batch

                if self.step % self.display_step_freq == 0:
                    tf.print("Step:", self.step,
                             "Dev acc:", self.evaluate_classifier(dev_data[0:1000]),
                             "Train acc:", self.evaluate_classifier(training_data[0:1000]),
                             "Avg Cost:", avg_cost,
                             "total step Cost :", total_cost)

                self.step += 1

            # Display some statistics about the step
            # Evaluating only one batch worth of data -- simplifies implementation slightly
            if self.epoch % self.display_epoch_freq == 0:
                dev_acc.append(self.evaluate_classifier(dev_data[0:1000]))
                train_acc.append(self.evaluate_classifier(training_data[0:1000]))
                epochs.append(self.epoch + 1)
                # improved the early stopping and the retrieving of the best model (params)
                # patience mechanism regarding the traceback of best dev metric
                if dev_acc[len(dev_acc) - 1] <= self.best_dev:
                    self.patience += 1
                    print(" {} epochs without improvement".format(self.patience))
                    if self.patience == self.max_patience:
                        print(" early stopping")
                        break  # early stop over dev curve
                else:
                    # track best features over training
                    # here we control the best parameters retrieval
                    self.best_dev = dev_acc[len(dev_acc) - 1]
                    self.best_params = self.trainable_variables
                    self.patience = 0

                tf.print("Epoch:", (self.epoch + 1),
                         "Dev acc:", self.evaluate_classifier(dev_data[0:1000]),
                         "Train acc:", self.evaluate_classifier(training_data[0:1000]),
                         "Avg Cost:", avg_cost,
                         "total step Cost :", total_cost)
            self.epoch += 1

        # retrieve best training features
        self.trainable_variables = self.best_params
        return train_acc, dev_acc, epochs

    def classify(self, examples):
        # This classifies a list of examples
        premise_vectors = np.vstack([example['reference_answer_index_sequence'] for example in examples])
        hypothesis_vectors = np.vstack([example['student_answer_index_sequence'] for example in examples])
        logits, _ = self.model(premise_vectors, hypothesis_vectors)
        return np.argmax(logits, axis=1)

    def get_attn(self, examples):
        premise_vectors = np.vstack([example['reference_answer_index_sequence'] for example in examples])
        hypothesis_vectors = np.vstack([example['student_answer_index_sequence'] for example in examples])
        _ , attn_weights, = self.model(premise_vectors, hypothesis_vectors)
        # target weights matrix
        return np.reshape(attn_weights, [len(examples), self.dim, self.dim])

    def get_minibatch(self, dataset, start_index, end_index):
        indices = range(start_index, end_index)
        premise_vectors = np.vstack([dataset[i]['reference_answer_index_sequence'] for i in indices])
        hypothesis_vectors = np.vstack([dataset[i]['student_answer_index_sequence'] for i in indices])
        labels = [dataset[i]['label'] for i in indices]
        return premise_vectors, hypothesis_vectors, labels

def plot_learning_curve(par_values, train_scores, dev_scores, model_path, title="Learning Curve", xlab="", ylab="Accuracy",ylim=None):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    par_values : list of checked values of the current parameter.

    train_scores : list of scores obtained in training set (same length as par_values).

    test_scores : list of scores obtained in dev set (same length as par_values)

    title : string
        Title for the chart.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    plt.grid()
    plt.plot(par_values, train_scores, color="r", label="Training score")
    plt.plot(par_values, dev_scores, color="g", label="Dev score")

    plt.legend(loc="best")
    plt.savefig(model_path + 'learning_curve.pdf')
    plt.clf()
    plt.cla()
    plt.close()

if not path.exists(model_path+"model.pickle"):
    print("model wasn't found, we currently generate new model")
    classifier = SimmilarityClassifier(len(word_indices), Sentence_len, ind2label)
    print(tf.__version__) # needed tensorflow 2.x
    train_acc, dev_acc, epochs = classifier.train(training_set, dev_set)# 10 epochs computed because is large model ...
    plot_learning_curve(epochs, train_acc, dev_acc, xlab="Epoch", model_path=model_path)
    print("Test acc:", classifier.evaluate_classifier(test_set))
    # save model
    pickle_out = open(model_path+"model.pickle","wb")
    pickle.dump(classifier.package_params(), pickle_out)
    # expected Test acc: 0.7894524959742351 (more less)
else:
    print("model was found, we currently loading the model")
    # load model
    pickle_in = open(model_path+"model.pickle", "rb")
    # loaded model, enabled reinforced learning ...
    classifier = SimmilarityClassifier(len(word_indices), Sentence_len, ind2label, test=True)
    classifier.setup_params(pickle.load(pickle_in))

    print("Test acc:", classifier.evaluate_classifier(test_set))
    print("Test set composed by {} examples".format(len(test_set)))

    # here we generate full understandable report (show the internal information) and check for the example
    example = list(training_set[:1])
    examples, autocorrelation = classifier.estimate_grade(example, output_figures_path, "1_exp")
    feedbacks = classifier.generate_assessment(examples, autocorrelation)
    for feedback in feedbacks:
        if feedback['estimated_score'] != feedback['score']:
            print("failed, tuning a bit ... please wait")
            # reinforced learning
            train_acc, dev_acc, epochs = classifier.train(training_set, test_set)  # 1 epochs ...
            train_acc, dev_acc, epochs = classifier.train(dev_set, test_set)

            example = list(training_set[:1])
            examples, autocorrelation = classifier.estimate_grade(example, output_figures_path, "1_exp")
            feedbacks = classifier.generate_assessment(examples, autocorrelation)

    class bcolors:
        OKGREEN = '\033[92m'
        FAIL = '\033[91m'

    for feedback in feedbacks:
        answer = feedback['answer']
        tokens = word_tokenize(answer)
        highligth_vec = feedback['highlight_vector']
        result = f""
        for n, token in enumerate(tokens):
            try:
                if highligth_vec[n] == 2:
                    result += str(bcolors.FAIL) + token + " "
                elif highligth_vec[n] == 1:
                    result += str(bcolors.OKGREEN) + token + " "
            except:
                pass

        print("\n")
        print("was estimated as: {}".format(feedback['estimated_score']))
        print("was graded as: {}".format(feedback['estimated_mark']))
        print("was really: {}".format(feedback['score']))
        print("\n")
        print("The feedback was:")
        print(feedback['feedback'])
        print("The reference was:")
        print(feedback['reference_answer'])
        print("\n")
        print("The question was:")
        print(feedback['question'])
        print("The answer was:")
        print(result)
