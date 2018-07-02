from urllib.parse import urlparse, parse_qs
from math import pow, log
import csv
import re
import matplotlib.pyplot as plt
import argparse
import os


class Parser:
    def __init__(self):

        self.train_dataset = ['em.csv', 'wp.csv']
        self.test_dataset = ['test_data.csv']
        self.feature_vector_file = ['feature_vector.csv']

        self.test = 0           # if not set that training mode else testing mode

        self.em = 0
        self.wp = 1

        self.domain_dict = {}
        self.query_dict = {}
        self.path_dict = {}
        self.parsed_domain_dict = {}

        self.dataset_length = [0, 0]

        self.flag = 1  # if flag is set then modified ig will be used else original ig is used

        self.threshold = {'domain': 0.69475, 'parsed_domain': 0.6927, 'query': 0.6927, 'path': 0.6927}
        self.feature_vector = {'domain': [], 'parsed_domain': [], 'query': [], 'path': []}

    def parse_path(self, path):
        path = path.lower()  # to lower case the url
        path = path.replace(".", "/")
        path = path.replace("-", "/")
        path = path.replace(":", "/")
        path = path.replace("&", "/")
        path = path.replace("%", "/")
        output = re.split('/', path)

        special_characters = "_!@#$%^*()_+><}{][';,"
        numbers = "0123456789"

        # TODO stopwords.txt and more unnecessary words and use regex

        unnecessary_word = ["http", "www", "https", "", "com"]
        for i in unnecessary_word:
            output = list(filter(lambda j: j != i, output))

        for i in range(len(output)):
            if output[i].isdigit():
                output[i] = "numb"
            elif any((c in special_characters) for c in output[i]):
                output[i] = "spechar"
            elif output[i].isalnum() and any((c in numbers) for c in output[i]):
                output[i] = "isalnum"
            else:
                continue

        temp_output = []
        for i in output:  # to remove words with length 1
            if len(i) == 1:
                temp_output.append(i)

        for i in temp_output:
            output = list(filter(lambda j: j != i, output))

        return output

    def url_length(self, url):

        url = url.lower()  # to lower case the url
        url = url.replace(".", "/")
        url = url.replace("-", "/")
        url = url.replace(":", "/")
        url = url.replace("&", "/")
        output = re.split('/', url)

        url_len = len(output)
        return url_len

    def prep_dict(self):
        for i in self.input_dataset:
            file_object = open(i)
            reader = csv.reader(file_object)
            for row in reader:

                # print(row)
                parsed_url_tuple = urlparse(row[5])
                parsed_query_dict = parse_qs(parsed_url_tuple.query)
                path_list = self.parse_path(parsed_url_tuple.path)
                domain_list = self.parse_path(parsed_url_tuple.netloc)

                if row[4] == "em":
                    index = self.em
                    new_list = [1, 0]
                    self.dataset_length[self.em] = self.dataset_length[self.em] + 1

                elif row[4] == "wp":
                    index = self.wp
                    new_list = [0, 1]
                    self.dataset_length[self.wp] = self.dataset_length[self.wp] + 1
                else:
                    continue

                if parsed_url_tuple.netloc in self.domain_dict:
                    self.domain_dict[parsed_url_tuple.netloc][index] = self.domain_dict[parsed_url_tuple.netloc][
                                                                           index] + 1
                    # print(self.domain_dict)
                else:
                    self.domain_dict[parsed_url_tuple.netloc] = list(new_list)

                # print(self.domain_dict)

                for args in list(parsed_query_dict):
                    if args in self.query_dict:
                        self.query_dict[args][index] = self.query_dict[args][index] + 1
                    else:
                        self.query_dict[args] = list(new_list)

                for token in path_list:
                    if token in self.path_dict:
                        self.path_dict[token][index] = self.path_dict[token][index] + 1
                    else:
                        self.path_dict[token] = list(new_list)

                for token in domain_list:
                    if token in self.parsed_domain_dict:
                        self.parsed_domain_dict[token][index] = self.parsed_domain_dict[token][index] + 1
                    else:
                        self.parsed_domain_dict[token] = list(new_list)

    def manually_prep_dataset(self):  # TODO must be in a seperate file, use command line arguments,update naming

        file_object = open("email_pixel_dataset6.csv")
        file_object_writer = open("joined_dataset.csv", "w")
        reader = csv.reader(file_object)
        writer = csv.writer(file_object_writer)

        em_list = []
        wp_list = []

        for row in reader:
            writer.writerow(row)
            em_list.append(row)
        file_object.close()  # todo write part in a funciton
        file_object = open("web_pixel_dataset2.csv")
        reader = csv.reader(file_object)
        for row in reader:
            writer.writerow(row)
            wp_list.append(row)

        file_object_writer.close()
        file_object.close()

        common_dict = dict(self.domain_dict_em)
        for key, value in self.domain_dict_wp.items():
            if key in common_dict:
                common_dict[key] = common_dict[key] + value
            else:
                common_dict[key] = value
        # sorted_dict = {}
        sorted_dict = dict([(k, common_dict[k]) for k in sorted(common_dict, key=common_dict.get, reverse=True)])

        domain_scanned = 0
        urls_scanned = 0
        #
        # print(sorted_dict)
        # total_url = 0
        # count = 0
        # for key, value in sorted_dict.items():
        #     total_url = total_url + value
        #     count = count + 1
        #     print(key,":",value)
        #     if count > 60:
        #         break

        #
        # print(total_url)

        for key, value in sorted_dict.items():

            print("domain scanned", domain_scanned)
            print("urls scanned", urls_scanned)
            domain_scanned = domain_scanned + 1
            urls_scanned = urls_scanned + value

            print(key, ":", value)
            choice = input("Choice :")

            if choice == 'e':
                for i in em_list:
                    if i[4] == 'wp' and urlparse(i[5]).netloc == key:
                        i[4] = 'em'
                for i in wp_list:
                    if i[4] == 'wp' and urlparse(i[5]).netloc == key:
                        i[4] = 'em'

            elif choice == 'w':
                for i in em_list:
                    if i[4] == 'em' and urlparse(i[5]).netloc == key:
                        i[4] = 'wp'
                for i in wp_list:
                    if i[4] == 'em' and urlparse(i[5]).netloc == key:
                        i[4] = 'wp'
            else:
                continue

            if domain_scanned % 5 == 0:
                exit_status = eval(input("you want to continue(for exit press 1)"))
                if exit_status == 1:
                    break;

            # object writer lines are commented(2 lines) as the em.csv and wp.csv contents must not be
            # modified.
            # file_object_writer = open("em.csv", "w")
            writer = csv.writer(file_object_writer)
            for i in em_list:
                writer.writerow(i)
            file_object_writer.close()

            # file_object_writer = open("wp.csv", "w")
            writer = csv.writer(file_object_writer)
            for i in wp_list:
                writer.writerow(i)
            file_object_writer.close()

    def calculate_information_gain(self, keyword_dict, entropy_y):

        for key, value in keyword_dict.items():
            total_freq = value[0] + value[1]
            if value[0] != 0:
                prob_c1 = float(value[0] / total_freq) * log(float(value[0] / total_freq))
                prob_inv_c1 = float((self.dataset_length[0] - value[0]) / self.dataset_length[0]) * log(
                    float((self.dataset_length[0] - value[0]) / self.dataset_length[0]))
            else:
                prob_c1 = 0
                prob_inv_c1 = 0

            if value[1] != 0:
                prob_c2 = float(value[1] / total_freq) * log(float(value[1] / total_freq))
                prob_inv_c2 = float((self.dataset_length[1] - value[1]) / self.dataset_length[1]) * log(
                    float((self.dataset_length[1] - value[1]) / self.dataset_length[1]))

            else:
                prob_c2 = 0
                prob_inv_c2 = 0

            if self.flag:
                keyword_dict.append(entropy_y + (prob_c1 + prob_c2) * ((value[0] + value[1]) / self.dataset_length[2]) \
                                    - (prob_inv_c1 + prob_inv_c2) * float(
                    (self.dataset_length[2] - total_freq) / self.dataset_length[2]))
            else:
                keyword_dict.append(
                    1 - (entropy_y + (prob_c1 + prob_c2) * ((value[0] + value[1]) / self.dataset_length[2])
                         + (prob_inv_c1 + prob_inv_c2) * float(
                                (self.dataset_length[2] - total_freq) / self.dataset_length[2])))

    def set_information_gain(self):
        self.dataset_length.append(self.dataset_length[0] + self.dataset_length[1])
        entropy_y = - (self.dataset_length[0] / self.dataset_length[2]) * log(
            self.dataset_length[0] / self.dataset_length[2]) \
                    - (self.dataset_length[1] / self.dataset_length[2]) * log(
            self.dataset_length[1] / self.dataset_length[2])

        self.calculate_information_gain(self.domain_dict, entropy_y)
        self.calculate_information_gain(self.query_dict, entropy_y)
        self.calculate_information_gain(self.path_dict, entropy_y)
        self.calculate_information_gain(self.parsed_domain_dict, entropy_y)

    def prepare_feature_vector(self):

        file_object = open("feature_vector.csv", "w")
        writer = csv.writer(file_object)

        for key, value in self.domain_dict.items():
            if value[2] > self.threshold['domain']:
                self.feature_vector['domain'].append(key)

        for key, value in self.parsed_domain_dict.items():
            if value[2] > self.threshold['parsed_domain']:
                self.feature_vector['parsed_domain'].append(key)

        for key, value in self.path_dict.items():
            if value[2] > self.threshold['path']:
                self.feature_vector['path'].append(key)

        for key, value in self.query_dict.items():
            if value[2] > self.threshold['query']:
                self.feature_vector['query'].append(key)

        writer.writerow(self.feature_vector['domain'])
        writer.writerow(self.feature_vector['parsed_domain'])
        writer.writerow(self.feature_vector['path'])
        writer.writerow(self.feature_vector['query'])

        file_object.close()

    def display(self):
        # print("length of the dataset", self.dataset_length)
        # print("domain dict", self.domain_dict)
        # print("query dict", self.query_dict)
        print("path dict", self.query_dict)
        # print("parsed domain dict", self.parsed_domain_dict)

    def plot_dict_graph(self):
        domain_list = []
        y = []
        index = 0
        for key, value in self.domain_dict.items():
            index = index + 1
            domain_list.append(value[2])
            y.append(index)

        plt.scatter(y, domain_list)
        plt.title("Domain")
        plt.show()

        y = []
        domain_list = []
        index = 0
        for key, value in self.query_dict.items():
            domain_list.append(value[2])
            index = index + 1
            y.append(index)

        plt.scatter(y, domain_list)
        plt.title("Query parameters")
        plt.show()

        y = []
        domain_list = []
        index = 0
        for key, value in self.parsed_domain_dict.items():
            domain_list.append(value[2])
            index = index + 1
            y.append(index)

        plt.scatter(y, domain_list)
        plt.title("Parsed Domain")
        plt.show()

        y = []
        domain_list = []
        index = 0
        for key, value in self.path_dict.items():
            domain_list.append(value[2])
            index = index + 1
            y.append(index)

        plt.scatter(y, domain_list)
        plt.title("Path")
        plt.show()

    def threshold_frequency(self):
        th_domain = 0.69475
        count = 0
        for key, value in self.domain_dict.items():
            if value[2] > th_domain:
                count = count + 1
        print("total domain with ig > ", th_domain, "are : ", count)
        print(len(self.domain_dict))
        th_parsed_domain = 0.6927
        count = 0
        for key, value in self.parsed_domain_dict.items():
            if value[2] > th_parsed_domain:
                count = count + 1
        print("total parsed domain with ig > ", th_parsed_domain, "are : ", count)
        print(len(self.parsed_domain_dict))
        th_query = 0.6927
        count = 0
        for key, value in self.query_dict.items():
            if value[2] > th_query:
                count = count + 1
        print("total queries with ig > ", th_query, "are : ", count)
        print(len(self.query_dict))
        th_path = 0.6927
        count = 0
        for key, value in self.path_dict.items():
            if value[2] > th_path:
                count = count + 1
        print("total path keywords with ig > ", th_path, "are : ", count)
        print(len(self.path_dict))


    def set_vector(self, input_list, feature_vector, vector):

        for token in input_list:
            if token in feature_vector:
                vector[feature_vector.index(token)] = 1
        return vector

    # function to read data from csv file based on self.test flag and then preprocess of dataset
    # the name of the outputfile if not passed will be preprocessed_file.csv
    def data_preprocessing(self, preprocessed_file="preprocessed_file.csv"):

        feature_file_object = open(self.feature_vector_file)
        feature_rows = csv.reader(feature_file_object)
        multiple_feature_rows = []
        for row in feature_rows:
            multiple_feature_rows.append(row)
        feature_file_object.close()

        self.feature_vector['domain'] = multiple_feature_rows[0]
        self.feature_vector['parsed_domain'] = multiple_feature_rows[1]
        self.feature_vector['path'] = multiple_feature_rows[2]
        self.feature_vector['query'] = multiple_feature_rows[3]

        # in case of training data open -> processed_data.csv
        # in case of test data open -> processed_data_test.csv

        file_object = open(preprocessed_file, "w")
        output_file_writer = csv.writer(file_object)

        # in case of training data loop over -> input_dataset
        if self.test:
            file_read = self.test_dataset
        else:
            file_read = self.train_dataset

        for i in file_read:
            file_object_reader = open(i)
            file_reader = csv.reader(file_object_reader)

            object = {}

            for row in file_reader:

                parsed_url_tuple = urlparse(row[5])
                parsed_query = list(parse_qs(parsed_url_tuple.query))
                path_list = self.parse_path(parsed_url_tuple.path)
                domain_list = self.parse_path(parsed_url_tuple.netloc)

                object['domain'] = [0] * len(self.feature_vector['domain'])
                if parsed_url_tuple.netloc in self.feature_vector['domain']:
                    object['domain'][self.feature_vector['domain'].index(parsed_url_tuple.netloc)] = 1

                object['query'] = [0] * len(self.feature_vector['query'])
                object['query'] = self.set_vector(parsed_query, self.feature_vector['query'], object['query'])

                object['path'] = [0] * len(self.feature_vector['path'])
                object['path'] = self.set_vector(path_list, self.feature_vector['path'], object['path'])

                object['parsed_domain'] = [0] * len(self.feature_vector['parsed_domain'])
                object['parsed_domain'] = self.set_vector(domain_list, self.feature_vector['parsed_domain'], object['parsed_domain'])

                if row[5].find('mail') == -1:
                    object['mail'] = 0
                else:
                    object['mail'] = 1

                object['url_length'] = self.url_length(row[5])

                if row[4] == 'em':
                    object['tag_type'] = 1
                else:
                    object['tag_type'] = 0

                write_row = [object['mail']] + object['domain'] + object['parsed_domain'] + object['path'] + object[
                    'query'] + [object['url_length']] + [object['tag_type']]
                output_file_writer.writerow(write_row)
            file_object_reader.close()
        file_object.close()


if __name__ == "__main__":

    cl_args = argparse.ArgumentParser()
    cl_args.add_argument("-path", help="checkpoint path")
    args = cl_args.parse_args()
    print(args.path)
    print()

    # parser = Parser()
    # parser.prep_dict()
    # parser.set_ig()
    # parser.prepare_feature_vector()
    # parser.data_transformation()
    # parser.threshold_frequency()
    # parser.display()
    # parser.plot_dict_graph()

