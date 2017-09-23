
emission_probability = {}
transition_probability = {}
bigram_tags_count = {}
bigram_word_tag_count = {}
unigram_tags_count = {}
unigram_tag = []
unigram_vocab_dict = {}
UNKNOWN_WORD = 'UNK'
ADD_K_SMOOTHING_CONSTANT = 0.75
bigram_tag_vocab = 0
unknown_word_tags = []


def populate_bigram_probability(input_unigram_file, input_bigram_file,
                                input_word_tag_file, input_word_file):
    """
    Populates emission and transition probabilities dictionary.
    :param input_unigram_file: File containing all unigram tags and their count.
    :param input_bigram_file: File containing all bigram tags and their count.
    :param input_word_tag_file: File containing all words and their tags.
    :param input_word_file: Fil containing all the unique words.
    """

    _populate_unigram_tags(input_unigram_file)
    _populate_bigram_tags(input_bigram_file)
    _calculate_transition_probability(unigram_tag, bigram_tags_count, input_bigram_file)
    _convert_unknown_words(input_word_file)
    _calculate_bigram_word_tag_counts(input_word_tag_file)
    _calculate_emission_probability(bigram_word_tag_count)


def _populate_unigram_tags(input_unigram_file):
    """
        Populates the unigram tag counts in a dictionary
    :param input_unigram_file:
    """
    with open(input_unigram_file, "r") as f:
        data = f.readlines()
        for line in data:
            words = line.split()

            unigram_tags_count[words[1]] = float(words[0])
            unigram_tag.append(words[1])


def _populate_bigram_tags(input_bigram_file):
    """
        Populates the bigram tag counts in dictionary
    :param input_bigram_file:
    :return:
    """
    with open(input_bigram_file, "r") as f:
        data = f.readlines()
        bigram_tag_vocab = len(f.readlines())
        for line in data:
            words = line.split()

            bigram_tags_count[words[1] + '|' + words[2]] = float(words[0])


def _calculate_transition_probability(unigram_tag, bigram_tags_count, input_bigram_file):
    """
        Populates those bigram tag counts, whose pairs don't exist in the training data as 0.
        Populates Transition Probability.
        The bigram probability is counted using the formula:
        P(w | w-1) = C(w-1 w)/ C(w-1)
    :param unigram_tag:
    :param bigram_tags_count:
    :return:
    """

    # for tag in range(0, len(unigram_tag)):
    #     for next_tag in range(0, len(unigram_tag)):
    #         if unigram_tag[tag] + '|' + unigram_tag[next_tag] not in bigram_tags_count:
    #             bigram_tags_count[unigram_tag[tag] + '|' + unigram_tag[next_tag]] = 0
    #         transition_probability[unigram_tag[next_tag] + '|' + unigram_tag[tag]] = \
    #             bigram_tags_count[unigram_tag[tag] + '|' + unigram_tag[next_tag]] / unigram_tags_count[unigram_tag[tag]]

    # Using Add k smoothingw with k = 0.75
    for tag in range(0, len(unigram_tag)):
        for next_tag in range(0, len(unigram_tag)):
            if unigram_tag[tag] + '|' + unigram_tag[next_tag] not in bigram_tags_count:
                bigram_tags_count[unigram_tag[tag] + '|' + unigram_tag[next_tag]] = 0
            transition_probability[unigram_tag[next_tag] + '|' + unigram_tag[tag]] = \
                (bigram_tags_count[unigram_tag[tag] + '|' + unigram_tag[next_tag]] + ADD_K_SMOOTHING_CONSTANT) / \
                  (unigram_tags_count[unigram_tag[tag]] + ADD_K_SMOOTHING_CONSTANT * bigram_tag_vocab)


def _convert_unknown_words(input_word_file):
    """
         Words which occur only once are treated as `UNK`.
    :param input_word_file:
    :return:
    """
    with open(input_word_file, "r") as f:
        data = f.readlines()
        for line in data:
            words = line.split()
            if words[0] == '1':
                words[1] = UNKNOWN_WORD
                if UNKNOWN_WORD not in unigram_vocab_dict:
                    unigram_vocab_dict[words[1]] = 1
                else:
                    unigram_vocab_dict[words[1]] += 1
            else:
                unigram_vocab_dict[words[1]] = float(words[0])


def _calculate_bigram_word_tag_counts(input_word_tag_file):
    """
        Populates bigram {word, tag} frequencies in `bigram_word_tag_count` dict
    :param input_word_tag_file:
    :return:
    """
    with open(input_word_tag_file, "r") as f:
        data = f.readlines()
        for line in data:
            words = line.split()

            if words[1] in unigram_vocab_dict:
                bigram_word_tag_count[words[1] + '|' + words[2]] = float(words[0])
            else:
                # word is UNK
                if '{}|{}'.format(UNKNOWN_WORD, words[2]) not in bigram_word_tag_count:
                    bigram_word_tag_count['{}|{}'.format(UNKNOWN_WORD, words[2])] = float(words[0])
                else:
                    bigram_word_tag_count['{}|{}'.format(UNKNOWN_WORD, words[2])] += float(words[0])


def _calculate_emission_probability(bigram_word_tag_count):
    """
        Calculates the emission probability
    :param bigram_word_tag_count:
    :return:
    """
    for key, count in bigram_word_tag_count.iteritems():
        keys = key.split('|')
        emission_probability['{}|{}'.format(keys[0], keys[1])] = float(count) / float(unigram_tags_count[keys[1]])


def get_unknown_word_tags():
    if not unknown_word_tags:
        for k, v in emission_probability.iteritems():
            keys = k.split('|')
            if keys[0] == UNKNOWN_WORD:
                unknown_word_tags.append(keys[1])

    return unknown_word_tags
