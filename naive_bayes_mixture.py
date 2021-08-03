# naive_bayes_mixture.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

import numpy as np
from collections import Counter

"""
This is the main entry point for Part 2 of this MP. You should only modify code
within this file for Part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

def naiveBayesMixture(train_set, train_labels, dev_set, bigram_lambda,unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    bigram_lambda - float between 0 and 1

    unigram_smoothing_parameter - Laplace smoothing parameter for unigram model (between 0 and 1)

    bigram_smoothing_parameter - Laplace smoothing parameter for bigram model (between 0 and 1)

    pos_prior - positive prior probability (between 0 and 1)
    """

    # Initialize a list to store the predicted development set labels
    dev_labels = []

    # Transform unigram training set data into bigram training set data
    bigram_train_set = []
    for i in range(len(train_set)):
        bigram_train_set.append([i + " " + j for i, j in zip(train_set[i], train_set[i][1:])])

    # Transform unigram dev set data into bigram dev set data
    bigram_dev_set = []
    for i in range(len(dev_set)):
        bigram_dev_set.append([i + " " + j for i, j in zip(dev_set[i], dev_set[i][1:])])

    #print("Bigram Training Set:")
    #print(bigram_training_set)
    #print()

    # *--Train Bigram Model--*

    # Initialize Counters to store the frequency of every word in Ham/Spam emails
    bigram_ham_word_counter = Counter()
    bigram_spam_word_counter = Counter()

    # Initialize dictionaries to store the probability of each word Ham/Spam emails
    bigram_ham_word_probability = {}
    bigram_spam_word_probability = {}


    # Populate the frequency of every word in Ham/Spam emails
    index = 0
    for label in train_labels:
        if label == 1:
            bigram_ham_word_counter.update(bigram_train_set[index])
        else:
            bigram_spam_word_counter.update(bigram_train_set[index])

        index += 1

    # Display frequency
    print("Ham Word Counter:")
    print(bigram_ham_word_counter.most_common(10))
    print()

    print("Spam Word Counter:")
    print(bigram_spam_word_counter.most_common(10))
    print()

    # Determine the total number of words in the Ham/Spam training email set
    bigram_ham_total_words = 0
    for word_frequency in bigram_ham_word_counter.values():
        bigram_ham_total_words += word_frequency

    bigram_spam_total_words = 0
    for word_frequency in bigram_spam_word_counter.values():
        bigram_spam_total_words += word_frequency

    # Display totals BEFORE Laplace smoothing
    print("Total Number of Words in Ham Emails BEFORE Laplace:")
    print(bigram_ham_total_words)
    print()

    print("Total Number of Words in Spam Emails BEFORE Laplace:")
    print(bigram_spam_total_words)
    print()

    # Add the words present in the developer set but absent in the ham set to the counter with a frequency of zero
    for email in range(len(bigram_dev_set)):
        for word in bigram_dev_set[email]:
            if word not in bigram_ham_word_counter:
                bigram_ham_word_counter.update([word])
                bigram_ham_word_counter.subtract([word])

    # Add the words present in the developer set but absent in the spam set to the counter with a frequency of zero
    for email in range(len(bigram_dev_set)):
        for word in bigram_dev_set[email]:
            if word not in bigram_spam_word_counter:
                bigram_spam_word_counter.update([word])
                bigram_spam_word_counter.subtract([word])

    # Display the ham counter after the addition of words with zero frequency
    ham_word_counter_length = len(bigram_ham_word_counter)
    print("Smallest Ham Word Frequency:")
    print(bigram_ham_word_counter[ham_word_counter_length - 1])
    print()

    # Display the spam counter after the addition of words with zero frequency
    spam_word_counter_length = len(bigram_spam_word_counter)
    print("Smallest Spam Word Frequency:")
    print(bigram_spam_word_counter[spam_word_counter_length - 1])
    print()

    # Copy ham word counter content into ham word probability dictionary
    bigram_ham_word_probability = bigram_ham_word_counter.copy()

    # Copy spam word counter content into spam word probability dictionary
    bigram_spam_word_probability = bigram_spam_word_counter.copy()

    # Display dictionaries before the addition of the Laplace smoothing constant
    print("Ham Word Probability BEFORE Laplace:")
    # print(bigram_ham_word_probability)
    print()

    print("Spam Word Probability BEFORE Laplace:")
    # print(bigram_spam_word_probability)
    print()

    # Apply Laplace smoothing
    for word in bigram_ham_word_probability:
        bigram_ham_word_probability[word] += bigram_smoothing_parameter

    for word in bigram_spam_word_probability:
        bigram_spam_word_probability[word] += bigram_smoothing_parameter

    # Display the dictionaries after the addition of the Laplace smoothing constant
    print("Laplace Constant:")
    print(bigram_smoothing_parameter)
    print()

    print("Ham Word Probability AFTER Laplace:")
    # print(ham_word_probability)
    print()

    print("Spam Word Probability AFTER Laplace:")
    # print(spam_word_probability)
    print()

    # Determine the total number of words after Laplace smoothing
    bigram_ham_word_total = sum(bigram_ham_word_probability.values())

    bigram_spam_word_total = sum(bigram_spam_word_probability.values())

    # Display totals AFTER Laplace smoothing
    print("Total Number of Words in Ham Emails AFTER Laplace:")
    print(bigram_ham_word_total)
    print()

    print("Total Number of Words in Spam Emails AFTER Laplace:")
    print(bigram_spam_word_total)
    print()

    # Determine each word's likelihood in ham/spam emails (logging the probabilities to avoid underflow)
    for word in bigram_ham_word_probability:
        bigram_ham_word_probability[word] = np.log((bigram_ham_word_probability[word]) / bigram_ham_word_total)

    for word in bigram_spam_word_probability:
        bigram_spam_word_probability[word] = np.log((bigram_spam_word_probability[word]) / bigram_spam_word_total)

    # Determine likelihood of ham/spam prior [i.e: log(P(Ham)) and log(P(Spam))]
    bigram_likelihood_of_ham = np.log(pos_prior)
    bigram_likelihood_of_spam = np.log(1.0 - pos_prior)

    # *--Train Unigram Model--*
    # Initialize Counters to store the frequency of every word in Ham/Spam emails
    unigram_ham_word_counter = Counter()
    unigram_spam_word_counter = Counter()

    # Initialize dictionaries to store the probability of each word Ham/Spam emails
    unigram_ham_word_probability = {}
    unigram_spam_word_probability = {}

    # Populate the frequency of every word in Ham/Spam emails
    index = 0
    for label in train_labels:
        if label == 1:
            unigram_ham_word_counter.update(train_set[index])
        else:
            unigram_spam_word_counter.update(train_set[index])

        index += 1

    # Display frequency
    print("Unigram Ham Word Counter:")
    print(unigram_ham_word_counter.most_common(10))
    print()

    print("Unigram Spam Word Counter:")
    print(unigram_spam_word_counter.most_common(10))
    print()

    # Determine the total number of words in the Ham/Spam training email set
    unigram_ham_total_words = 0
    for word_frequency in unigram_ham_word_counter.values():
        unigram_ham_total_words += word_frequency

    unigram_spam_total_words = 0
    for word_frequency in unigram_spam_word_counter.values():
        unigram_spam_total_words += word_frequency

    # Display totals BEFORE Laplace smoothing
    print("Unigram: Total Number of Words in Ham Emails BEFORE Laplace:")
    print(unigram_ham_total_words)
    print()

    print("Unigram: Total Number of Words in Spam Emails BEFORE Laplace:")
    print(unigram_spam_total_words)
    print()

    # Add the words present in the developer set but absent in the ham set to the counter with a frequency of zero
    for email in range(len(dev_set)):
        for word in dev_set[email]:
            if word not in unigram_ham_word_counter:
                unigram_ham_word_counter.update([word])
                unigram_ham_word_counter.subtract([word])

    # Add the words present in the developer set but absent in the spam set to the counter with a frequency of zero
    for email in range(len(dev_set)):
        for word in dev_set[email]:
            if word not in unigram_spam_word_counter:
                unigram_spam_word_counter.update([word])
                unigram_spam_word_counter.subtract([word])

    # Display the ham counter after the addition of words with zero frequency
    unigram_ham_word_counter_length = len(unigram_ham_word_counter)
    print("Unigram: Smallest Ham Word Frequency:")
    print(unigram_ham_word_counter[unigram_ham_word_counter_length - 1])
    print()

    # Display the spam counter after the addition of words with zero frequency
    unigram_spam_word_counter_length = len(unigram_spam_word_counter)
    print("Unigram: Smallest Spam Word Frequency:")
    print(unigram_spam_word_counter[unigram_spam_word_counter_length - 1])
    print()

    # Copy ham word counter content into ham word probability dictionary
    unigram_ham_word_probability = unigram_ham_word_counter.copy()

    # Copy spam word counter content into spam word probability dictionary
    unigram_spam_word_probability = unigram_spam_word_counter.copy()

    # Display dictionaries before the addition of the Laplace smoothing constant
    print("Unigram: Ham Word Probability BEFORE Laplace:")
    # print(unigram_ham_word_probability)
    print()

    print("Unigram: Spam Word Probability BEFORE Laplace:")
    # print(unigram_spam_word_probability)
    print()

    # Apply Laplace smoothing
    for word in unigram_ham_word_probability:
        unigram_ham_word_probability[word] += unigram_smoothing_parameter

    for word in unigram_spam_word_probability:
        unigram_spam_word_probability[word] += unigram_smoothing_parameter

    # Display the dictionaries after the addition of the Laplace smoothing constant
    print("Unigram Laplace Constant:")
    print(unigram_smoothing_parameter)
    print()

    print("Unigram: Ham Word Probability AFTER Laplace:")
    # print(unigram_ham_word_probability)
    print()

    print("Unigram: Spam Word Probability AFTER Laplace:")
    # print(spam_word_probability)
    print()

    # Determine the total number of words after Laplace smoothing
    unigram_ham_word_total = sum(unigram_ham_word_probability.values())

    unigram_spam_word_total = sum(unigram_spam_word_probability.values())

    # Display totals AFTER Laplace smoothing
    print("Unigram: Total Number of Words in Ham Emails AFTER Laplace:")
    print(unigram_ham_word_total)
    print()

    print("Unigram: Total Number of Words in Spam Emails AFTER Laplace:")
    print(unigram_spam_word_total)
    print()

    # Determine each word's likelihood in ham/spam emails (logging the probabilities to avoid underflow)
    for word in unigram_ham_word_probability:
        unigram_ham_word_probability[word] = np.log((unigram_ham_word_probability[word]) / unigram_ham_word_total)

    for word in unigram_spam_word_probability:
        unigram_spam_word_probability[word] = np.log((unigram_spam_word_probability[word]) / unigram_spam_word_total)

    # Determine likelihood of ham/spam prior [i.e: log(P(Ham)) and log(P(Spam))]
    unigram_likelihood_of_ham = np.log(pos_prior)
    unigram_likelihood_of_spam = np.log(1.0 - pos_prior)


    # *----Test Model----*

    likelihood_email_is_ham = 0
    likelihood_email_is_spam = 0

    unigram_likelihood_email_is_ham = unigram_likelihood_of_ham
    unigram_likelihood_email_is_spam = unigram_likelihood_of_spam

    bigram_likelihood_email_is_ham = bigram_likelihood_of_ham
    bigram_likelihood_email_is_spam = bigram_likelihood_of_spam

    for email in range(len(dev_set)):
        # Based on the UNIGRAMS in a given email, determine the likelihood the email is ham and spam
        for unigram in dev_set[email]:
            unigram_likelihood_email_is_ham += unigram_ham_word_probability[unigram]
            unigram_likelihood_email_is_spam += unigram_spam_word_probability[unigram]

        # Based on the BIGRAMS in a given email, determine the likelihood the email is ham and spam
        for bigram in bigram_dev_set[email]:
            bigram_likelihood_email_is_ham += bigram_ham_word_probability[bigram]
            bigram_likelihood_email_is_spam += bigram_spam_word_probability[bigram]

        likelihood_email_is_ham = (1 - bigram_lambda) * (unigram_likelihood_email_is_ham) + (bigram_lambda) * (bigram_likelihood_email_is_ham)
        likelihood_email_is_spam = (1 - bigram_lambda) * (unigram_likelihood_email_is_spam) + (bigram_lambda) * (bigram_likelihood_email_is_spam)

        # Classify email as ham or spam based on likelihood value
        if likelihood_email_is_ham > likelihood_email_is_spam:
            dev_labels.append(1)
        else:
            dev_labels.append(0)

        # Reset likelihoods to initial values
        likelihood_email_is_ham = 0
        likelihood_email_is_spam = 0

        unigram_likelihood_email_is_ham = unigram_likelihood_of_ham
        unigram_likelihood_email_is_spam = unigram_likelihood_of_spam

        bigram_likelihood_email_is_ham = bigram_likelihood_of_ham
        bigram_likelihood_email_is_spam = bigram_likelihood_of_spam

    print("Bigram + Unigram Development Labels:")
    print(dev_labels)
    print()

    # return predicted labels of development set
    return dev_labels