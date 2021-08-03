# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

import numpy as np
from collections import Counter

"""
This is the main entry point for Part 1 of this MP. You should only modify code
within this file for Part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - positive prior probability (between 0 and 1)
    """

    # *----Train Model----*
    # Initialize Counters to store the frequency of every word in Ham/Spam emails
    ham_word_counter = Counter()
    spam_word_counter = Counter()

    # Initialize dictionaries to store the probability of each word Ham/Spam emails
    ham_word_probability = {}
    spam_word_probability = {}

    # Initialize a list to store the predicted development set labels
    dev_labels = []

    # Populate the frequency of every word in Ham/Spam emails
    index = 0
    for label in train_labels:
        if label == 1:
            ham_word_counter.update(train_set[index])
        else:
            spam_word_counter.update(train_set[index])

        index += 1

    # Display frequency
    print("Ham Word Counter:")
    print(ham_word_counter.most_common(10))
    print()

    print("Spam Word Counter:")
    print(spam_word_counter.most_common(10))
    print()

    # Determine the total number of words in the Ham/Spam training email set
    ham_total_words = 0
    for word_frequency in ham_word_counter.values():
        ham_total_words += word_frequency

    spam_total_words = 0
    for word_frequency in spam_word_counter.values():
        spam_total_words += word_frequency

    # Display totals BEFORE Laplace smoothing
    print("Total Number of Words in Ham Emails BEFORE Laplace:")
    print(ham_total_words)
    print()

    print("Total Number of Words in Spam Emails BEFORE Laplace:")
    print(spam_total_words)
    print()

    # Add the words present in the developer set but absent in the ham set to the counter with a frequency of zero
    for email in range(len(dev_set)):
        for word in dev_set[email]:
            if word not in ham_word_counter:
                ham_word_counter.update([word])
                ham_word_counter.subtract([word])

    # Add the words present in the developer set but absent in the spam set to the counter with a frequency of zero
    for email in range(len(dev_set)):
        for word in dev_set[email]:
            if word not in spam_word_counter:
                spam_word_counter.update([word])
                spam_word_counter.subtract([word])

    # Display the ham counter after the addition of words with zero frequency
    ham_word_counter_length = len(ham_word_counter)
    print("Smallest Ham Word Frequency:")
    print(ham_word_counter[ham_word_counter_length - 1])
    print()

    # Display the spam counter after the addition of words with zero frequency
    spam_word_counter_length = len(spam_word_counter)
    print("Smallest Spam Word Frequency:")
    print(spam_word_counter[spam_word_counter_length - 1])
    print()

    # Copy ham word counter content into ham word probability dictionary
    ham_word_probability = ham_word_counter.copy()

    # Copy spam word counter content into spam word probability dictionary
    spam_word_probability = spam_word_counter.copy()

    # Display dictionaries before the addition of the Laplace smoothing constant
    print("Ham Word Probability BEFORE Laplace:")
    # print(ham_word_probability)
    print()

    print("Spam Word Probability BEFORE Laplace:")
    # print(spam_word_probability)
    print()

    # Apply Laplace smoothing
    for word in ham_word_probability:
        ham_word_probability[word] += smoothing_parameter

    for word in spam_word_probability:
        spam_word_probability[word] += smoothing_parameter

    # Display the dictionaries after the addition of the Laplace smoothing constant
    print("Laplace Constant:")
    print(smoothing_parameter)
    print()

    print("Ham Word Probability AFTER Laplace:")
    # print(ham_word_probability)
    print()

    print("Spam Word Probability AFTER Laplace:")
    # print(spam_word_probability)
    print()

    # Determine the total number of words after Laplace smoothing
    ham_word_total = sum(ham_word_probability.values())

    spam_word_total = sum(spam_word_probability.values())

    # Display totals AFTER Laplace smoothing
    print("Total Number of Words in Ham Emails AFTER Laplace:")
    print(ham_word_total)
    print()

    print("Total Number of Words in Spam Emails AFTER Laplace:")
    print(spam_word_total)
    print()

    # Determine each word's likelihood in ham/spam emails (logging the probabilities to avoid underflow)
    for word in ham_word_probability:
        ham_word_probability[word] = np.log((ham_word_probability[word]) / ham_word_total)

    for word in spam_word_probability:
        spam_word_probability[word] = np.log((spam_word_probability[word]) / spam_word_total)

    # Determine likelihood of ham/spam prior [i.e: log(P(Ham)) and log(P(Spam))]
    likelihood_of_ham = np.log(pos_prior)
    likelihood_of_spam = np.log(1.0 - pos_prior)

    # *----Test Model----*
    likelihood_email_is_ham = likelihood_of_ham
    likelihood_email_is_spam = likelihood_of_spam

    for email in range(len(dev_set)):
        # Based on the words in a given email, determine the likelihood the email is ham and spam
        for word in dev_set[email]:
            likelihood_email_is_ham += ham_word_probability[word]
            likelihood_email_is_spam += spam_word_probability[word]

        # Classify email as ham or spam based on likelihood value
        if likelihood_email_is_ham > likelihood_email_is_spam:
            dev_labels.append(1)
        else:
            dev_labels.append(0)

        # Reset likelihoods to initial values
        likelihood_email_is_ham = likelihood_of_ham
        likelihood_email_is_spam = likelihood_of_spam

    print("Development Labels:")
    print(dev_labels)
    print()

    # return predicted labels of development set
    return dev_labels
