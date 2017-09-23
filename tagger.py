from constant import DOC_PATH, TRAINING_DATA_FILE, INPUT_PATH, OUTPUT_PATH, TEST_DATA_FILE, \
    UNIGRAM_TAG_FILE, BIGRAM_TAG_FILE, WORD_TAG_FILE, WORD_FILE
from data import emission_probability, transition_probability, populate_bigram_probability, bigram_word_tag_count,\
    unigram_vocab_dict, bigram_tags_count, unigram_tags_count


class Reader:
    """
        Reads file of the following format,
        Sentences arranged as one word per line with a blank line separating the sentences.
        The columns, tab separated, with the first column giving word position,
        the second the word and the third the POS tag.
    """

    def __init__(self, input_file):
        self._input_file = input_file
        self.word_map = {}
        self._reader()

    def get_word_map(self):
        """
            Returns the frequency word mapper, with keys as words and
            their tag frequencies as values
        :return: {Map} Frequency Word Mapper.
        """
        return self.word_map

    def _reader(self):
        """
        Reads the input file and stores the parsed data in `word map`.
        """
        with open(self._input_file, "r") as f:
            data = f.readlines()

            for line in data:
                words = line.split()

                if words:
                    if words[1] not in self.word_map:
                        self.word_map[words[1]] = {words[2]: 1}
                    else:
                        if not words[2] in self.word_map[words[1]]:
                            self.word_map[words[1]][words[2]] = 1
                        else:
                            self.word_map[words[1]][words[2]] += 1


class Writer:
    """
        Converts word map to (a file) generate taggers for each word in a sentence in the following format,
        Sentences arranged as one word per line with a blank line separating the sentences.
        The columns, tab separated, with the first column giving word position,
        the second the word and the third the POS tag.
    """
    def __init__(self, word_map, input_file, output_path):
        self._word_map = word_map
        self.input_file = input_file
        self.output_path = output_path
        self._writer()

    def _writer(self):
        lines_to_write = []

        # Reads the file to find the most frequent tag present and chooses that tag for the word.
        # if there does not exist a word in the training data, choose `NN` as the tag.
        with open(self.input_file, "r") as f:
            data = f.readlines()

            for line in data:
                tag_chosen = 'NA'
                words = line.split()
                if words:
                    if words[1] in self._word_map:
                        maximum_count = 0

                        for tag, count in self._word_map[words[1]].iteritems():
                            if int(count) > maximum_count:
                                maximum_count = int(count)
                                tag_chosen = tag
                    lines_to_write.append(line.strip() + '\t' + tag_chosen + '\n')
                else:
                    lines_to_write.append('\n')

        with open(self.output_path, 'w') as of:
            of.writelines(lines_to_write)


if __name__ == "__main__":
    training_data_file_path = DOC_PATH + TRAINING_DATA_FILE
    reader = Reader(training_data_file_path)
    word_map = reader.get_word_map()
    writer = Writer(word_map, input_file=INPUT_PATH+TEST_DATA_FILE,
                    output_path=OUTPUT_PATH+'output.txt')

    populate_bigram_probability(input_unigram_file=DOC_PATH + UNIGRAM_TAG_FILE,
                                input_bigram_file=DOC_PATH + BIGRAM_TAG_FILE,
                                input_word_tag_file=DOC_PATH + WORD_TAG_FILE,
                                input_word_file=DOC_PATH + WORD_FILE)

    print bigram_tags_count
    print unigram_tags_count
    print transition_probability