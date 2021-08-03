# tf_idf_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020
# Modified by Kiran Ramnath 02/13/2021

"""
This is the main entry point for the Extra Credit Part of this MP. You should only modify code
within this file for the Extra Credit Part -- the unrevised staff files will be used when your
code is evaluated, so be careful to not modify anything else.
"""

import numpy as np
import math
from collections import Counter, defaultdict
import time
import operator


def compute_tf_idf(train_set, train_labels, dev_set):
    """
    train_set - List of list of words corresponding with each mail
    example: suppose I had two mails 'like this city' and 'get rich quick' in my training set
    Then train_set := [['like','this','city'], ['get','rich','quick']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two mails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each mail that we are testing on
              It follows the same format as train_set

    Return: A list containing words with the highest tf-idf value from the dev_set documents
            Returned list should have same size as dev_set (one word from each dev_set document)
    """

    tfidf_score = []

    '''
    Tf
    '''
    # Number of times word w appears in Doc A
    frequency_of_words_in_each_dev_email = []

    for email in range(len(dev_set)):
        frequency_of_words_in_each_dev_email.append(Counter(dev_set[email]))

    #print("Frequency of Words In Each Dev Email:", frequency_of_words_in_each_dev_email)

    # Total number of words in Doc A
    total_number_of_words_per_dev_email = []

    for email in range(len(frequency_of_words_in_each_dev_email)):
        total_number_of_words_per_dev_email.append(sum(frequency_of_words_in_each_dev_email[email].values()))

    #print("Total Number of Words Per Dev Email:", total_number_of_words_per_dev_email)

    '''
    idf
    '''
    # Total number of docs in training set
    total_number_of_docs = len(train_set)
    #print("Total Number of Emails In Training Set:", total_number_of_docs)

    # 1 + Number of docs in training set containing word w
    frequency_of_words_in_each_training_email = []

    for email in range(len(train_set)):
        frequency_of_words_in_each_training_email.append(Counter(train_set[email]))

    #print("Frequency of Words In Each Training Email:", frequency_of_words_in_each_training_email)

    number_of_emails_per_word = defaultdict(lambda: 0)
    email_counter = 0

    for email in range(len(frequency_of_words_in_each_dev_email)):
        for word in frequency_of_words_in_each_dev_email[email]:
            for email in range(len(frequency_of_words_in_each_training_email)):
                if (frequency_of_words_in_each_training_email[email][word] > 0):
                    email_counter += 1

            number_of_emails_per_word[word] = email_counter
            email_counter = 0

    #print("Number of Emails with Word:", number_of_emails_per_word)

    '''
    Calculating tdidf
    '''

    highest_tdidf_words_per_email = {}

    for email in range(len(frequency_of_words_in_each_dev_email)):
        #print("**********************************************************************")
        #print("Email #", email)
        for word in frequency_of_words_in_each_dev_email[email]:
            #print("Word:", word)

            number_of_times_word_appears_in_doc_A = frequency_of_words_in_each_dev_email[email][word]
            total_number_words_in_doc_A = total_number_of_words_per_dev_email[email]

            total_number_of_docs_in_training_set = total_number_of_docs
            number_of_docs_in_training_set_containing_word = number_of_emails_per_word[word]

            tfidf = (number_of_times_word_appears_in_doc_A / total_number_words_in_doc_A) * np.log(
                total_number_of_docs_in_training_set / (1 + number_of_docs_in_training_set_containing_word))

            #print()
            #print("Number Of Times Word Appears in Doc A:", number_of_times_word_appears_in_doc_A)
            #print("Total Number of Word in Doc A:", total_number_words_in_doc_A)
            #print("Total Number of Docs in Training Set:", total_number_of_docs_in_training_set)
            #print("Number of Docs in Training Set Containing", word, ":",
            #      number_of_docs_in_training_set_containing_word)
            #print()
            #print("tfidf:", tfidf)
            #print("------------------------------------------------------")

            highest_tdidf_words_per_email[word] = tfidf

        tfidf_score.append(list(highest_tdidf_words_per_email.keys())[
                               list(highest_tdidf_words_per_email.values()).index(
                                   max(highest_tdidf_words_per_email.values()))])
        highest_tdidf_words_per_email.clear()


    print("tfidf score:", tfidf_score)

    # return list of words (should return a list, not numpy array or similar)
    return tfidf_score