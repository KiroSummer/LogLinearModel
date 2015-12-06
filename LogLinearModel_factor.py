#!/usr/bin/python
#coding=utf-8
import datetime
import math

class sentence:
    def __init__(self):
        self.word = []
        self.tag = []
        self.wordchars = []

class dataset: 
    def __init__(self): 
        self.sentences = []
        self.name = ""
        self.total_word_count = 0
    
    def open_file(self, inputfile):
        self.inputfile = open(inputfile, mode = 'r')
        self.name = inputfile.split('.')[0]

    def close_file(self):
        self.inputfile.close()

    def read_data(self, sentenceLen):
        wordCount = 0
        sentenceCount = 0
        sen = sentence()
        for s in self.inputfile:
            if(s == '\n' or s == '\r\n'):
                sentenceCount += 1
                self.sentences.append(sen)
                sen = sentence()
                if(sentenceLen !=-1 and sentenceCount >= sentenceLen):
                    break
                continue
            list_s = s.split('\t')
            str_word = list_s[1].decode('utf-8')
            str_tag = list_s[3]
            list_wordchars = list(str_word)
            sen.word.append(str_word)
            sen.tag.append(str_tag)
            sen.wordchars.append(list_wordchars)
            wordCount += 1
        self.total_word_count = wordCount
        print(self.name + ".conll contains " + str(len(self.sentences)) + " sentences")
        print(self.name + ".conll contains " + str(self.total_word_count) + " words")

class log_linear_model:
    def __init__(self):
        self.feature = dict()
        self.feature_keys = []
        self.feature_values = []
        self.feature_length = 0
        self.feature_space_length = 0
        self.tags = dict()
        self.tags_length = 0
        self.w = []
        self.w_update_times = []
        self.w_all_eta = []
        self.g = []
        self.g_update_id = dict()
        self.train = dataset()
        self.dev = dataset()

        self.train.open_file("train.conll")
        #self.train.open_file("./bigdata/train.conll")
        self.train.read_data(-1)
        self.train.close_file()

        self.dev.open_file("dev.conll")
        #self.dev.open_file("./bigdata/dev.conll")
        self.dev.read_data(-1)
        self.dev.close_file()
    
    def create_feature(self, sentence, pos):
        word_count = len(sentence.word)
        wi = sentence.word[pos]
        pos_word_len = len(sentence.word[pos])
        if(pos == 0):
            wi_left_word = "START"
            wi_left_word_last_c = "T"
        else:
            wi_left_word = sentence.word[pos-1]
            wi_left_word_last_c = sentence.wordchars[pos-1][len(sentence.word[pos-1])-1]
        if(pos == word_count-1):
            wi_right_word = "END"
            wi_right_word_first_c = "E"
        else:
            wi_right_word = sentence.word[pos+1]
            wi_right_word_first_c = sentence.wordchars[pos+1][0]
        wi_last_c = sentence.wordchars[pos][pos_word_len - 1]
        wi_first_c = sentence.wordchars[pos][0]
        f = []
        f.append("02:" + wi)
        f.append("03:" + wi_left_word)
        f.append("04:" + wi_right_word)
        f.append("05:" + wi + '*' + wi_left_word_last_c)
        f.append("06:" + wi + '*' + wi_right_word_first_c)
        f.append("07:" + wi_first_c)
        f.append("08:" + wi_last_c)
        for i in range(1, pos_word_len - 1):
            wi_kth_c = sentence.wordchars[pos][i]
            f.append("09:" + wi_kth_c)
            f.append("10:" + wi_first_c + "*" + wi_kth_c)
            f.append("11:" + wi_last_c + "*" + wi_kth_c)
        for i in range(0, pos_word_len - 1):
            wi_kth_c = sentence.wordchars[pos][i]
            wi_kth_next_c = sentence.wordchars[pos][i + 1]
            if(wi_kth_c == wi_kth_next_c):
                f.append("13:" + wi_kth_c + "*" + "consecutive")
        if(pos_word_len == 1):
            f.append("12:" + wi + "*" + wi_left_word_last_c + "*" + wi_right_word_first_c)
        for i in range(0, pos_word_len):
            if(i >= 4):
                break
            f.append("14:" + sentence.word[pos][0:(i + 1)])
            f.append("15:" + sentence.word[pos][-(i + 1)::])
        return f

    def create_feature_space(self):
        feature_index = 0
        tag_index = 0
        for s in self.train.sentences:
            for p in range(0, len(s.word)):
                tag = s.tag[p]
                f = self.create_feature(s, p)
                for feature in f:    #创建特征空间
                    if (feature in self.feature):
                        pass
                    else:
                        self.feature[feature] = feature_index
                        feature_index += 1
                if(tag in self.tags):
                    pass
                else:
                    self.tags[tag] = tag_index
                    tag_index += 1
        self.feature_length = len(self.feature)
        self.tags_length = len(self.tags)
        self.feature_space_length = self.feature_length * self.tags_length
        self.w = [0] * self.feature_space_length
        self.w_update_times = [0] * self.feature_space_length
        self.g = [0] * self.feature_space_length
        self.feature_keys = list(self.feature.keys())
        self.feature_values = list(self.feature.values())
        print("the total number of features is " + str(self.feature_length))
        print("the total number of tags is " + str(self.tags_length))
        print("the length of the feature space is " + str(self.feature_space_length))

    def dot(self, f_id, offset):
        score = 0.0
        for f in f_id:
            score += self.w[offset + f]
        return score

    def get_feature_id(self, fv):
        fv_id = []
        for feature in fv:
            if(feature in self.feature):
                fv_id.append(self.feature[feature])
        return fv_id;
	
    def max_tag(self, sentence, pos):
        maxscore = -1.0
        tempscore = 0.0
        tag = "NULL"
        fv = self.create_feature(sentence, pos)
        fv_id = self.get_feature_id(fv)
        for t in self.tags:
            tempscore = self.dot(fv_id, self.feature_length * self.tags[t])
            if(tempscore > (maxscore + 1e-10)):
                maxscore = tempscore
                tag = t
        return tag

    def update_g(self, s, p, update_times):
        denominator = 0.0
        feature = self.create_feature(s, p)
        word = s.word[p]
        feature_id = self.get_feature_id(feature)
        correcttag_id = self.tags[s.tag[p]] 
        for i in feature_id:    #g加上正确的tag所对应的向量
            index = self.feature_length * correcttag_id + i
            self.g[index] += 1.0
            self.g_update_id[index] = 0
        for tag in self.tags:    #得到分母
            tag_id  = self.tags[tag]
            offset = self.feature_length * tag_id
            denominator += math.e ** self.dot(feature_id, offset)
        for tag in self.tags:
            currenttag_id = self.tags[tag]
            offset = self.feature_length * currenttag_id
            probability = 1.0 * (math.e ** self.dot(feature_id, offset)) / denominator    #每一个tag对应的概率
            #print("update_times:\t"+str(update_times)+"\tprobability:\t"+str(probability))
            for i in feature_id:
                index = offset + i
                self.g[index] -= probability * 1.0
                self.g_update_id[index] = 0

    def update_weight(self, eta, update_times):
        c = 0.01
        for i in range(self.feature_space_length):
            self.w[i] = (1 - eta * c) * self.w[i] + eta * self.g[i]
            #self.w[i] += eta * self.g[i]
            #self.w[i] = (1 - eta) * self.w[i] + self.g[i]

    def online_training(self):
        max_train_precision = 0.0
        max_dev_precision = 0.0
        B = 50
        b = 0
        eta = 0.01
        self.w_all_eta.append(eta)
        update_times = 0
        print("eta is " + str(eta))
        for iterator in range(0, 20):
            print("iterator " + str(iterator))
            for s in self.train.sentences:
                for p in range(0, len(s.word)):
                    self.update_g(s, p, update_times)
                    b += 1
                    if(B == b):
                        update_times += 1
                        self.update_weight(eta, update_times)
                        b = 0
                        #eta = max(eta * 0.999, 0.00001)
                        self.w_all_eta.append(eta)
                        self.g = [0] * self.feature_space_length
                        self.g_update_id.clear()
            if(b != 0):
                update_times += 1
            self.update_weight(eta, update_times)
            b = 0
            #eta = max(eta * 0.999, 0.00001)
            self.w_all_eta.append(eta)
            self.g = [0] * self.feature_space_length
            self.g_update_id.clear()

            self.save_model(iterator)
            #进行评估
            train_iterator, train_c, train_count, train_precision = self.evaluate(self.train, iterator)
            dev_iterator, dev_c, dev_count, dev_precision = self.evaluate(self.dev, iterator)
            #保存概率最大的情况
            if(train_precision > (max_train_precision + 1e-10)):
                max_train_precision = train_precision
                max_train_iterator = train_iterator
                max_train_c = train_c
                max_train_count = train_count
            if(dev_precision > (max_dev_precision + 1e-10)):
                max_dev_precision = dev_precision
                max_dev_iterator = dev_iterator
                max_dev_c = dev_c
                max_dev_count  = dev_count
        print("Conclusion:")
        print("\t"+self.train.name + " iterator: "+str(max_train_iterator)+"\t"+str(max_train_c)+" / "+str(max_train_count) + " = " +str(max_train_precision))
        print("\t"+self.dev.name + " iterator: "+str(max_dev_iterator)+"\t"+str(max_dev_c)+" / "+str(max_dev_count) + " = " +str(max_dev_precision))

    def save_model(self, iterator):
        fmodel = open("linearmodel.lm"+str(iterator), mode='w')
        for feature_id in self.feature_values:
            feature = self.feature_keys[feature_id]
            left_feature = feature.split(':')[0] + ':'
            right_feature = '*' + feature.split(':')[1]
            for tag in self.tags:
                tag_id = self.tags[tag]
                entire_feature = left_feature + tag + right_feature
                w = self.w[tag_id * self.feature_length + feature_id]
                if(w != 0):
                    fmodel.write(entire_feature.encode('utf-8') + '\t' + str(w) + '\n')
        fmodel.close()

    def evaluate(self, dataset, iterator):
       c = 0
       count = 0
       fout = open(dataset.name+".out" + str(iterator), mode='w')
       for s in dataset.sentences:
           for p in range(0, len(s.word)):
               count += 1
               max_tag = self.max_tag(s, p)
               correcttag = s.tag[p]
               fout.write(s.word[p].encode('utf-8') + '\t' + str(max_tag) + '\t' + str(correcttag) + '\n')
               if(max_tag != correcttag):
                   pass
               else:
                   c += 1
       print(dataset.name + "\tprecision is " + str(c) + " / " + str(count) + " = " + str(1.0 * c/count))
       fout.close()
       return iterator, c, count, 1.0 * c/count     


################################ main #####################################
if __name__ == '__main__':
    starttime = datetime.datetime.now()
    llm = log_linear_model()
    llm.create_feature_space()
    llm.online_training()
    endtime = datetime.datetime.now()
    print("executing time is "+str((endtime-starttime).seconds)+" s")
