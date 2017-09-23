class Viterbi:
    def __init__(self, tag_set, word_set,
                 transition_probability, emission_probability):
        self.tag_set = tag_set
        self.word_set = word_set
        self.transition_probability = transition_probability
        self.emission_probability = emission_probability
        self.viterbi = {}
        self.backtrack = {}
        self.states = []

    def get_viterbi_states(self):
        return self._viterbi()

    def _viterbi(self):
        """
            Implements viterbi algorithm to get sequence of tags associated with a sentence.
        :return: sequence of tags (states)
        """

        # Initialization
        for tag in self.tag_set:
            if not '{}|{}'.format(self.word_set[0], tag) in self.emission_probability:
                self.viterbi[tag] = {0: 0.0}
            else:
                self.viterbi[tag] = {0: float(self.transition_probability['{}|{}'.format(tag, '.')]) *
                                      float(self.emission_probability['{}|{}'.format(self.word_set[0], tag)])}

            self.backtrack[tag] = []

        # Populates Viterbi Matrix
        for word in range(1, len(self.word_set)):
            for tag in self.tag_set:
                max_probability = 0.0
                backtrack = tag
                for prev_tag in self.tag_set:
                    current_tag_probability = float(self.viterbi[prev_tag][word - 1]) * \
                                              float(self.transition_probability['{}|{}'.format(tag, prev_tag)])
                    if max_probability < current_tag_probability:
                        max_probability = current_tag_probability
                        backtrack = prev_tag

                if not '{}|{}'.format(self.word_set[word], tag) in self.emission_probability:
                    self.viterbi[tag][word] = 0
                else:
                    self.viterbi[tag][word] = max_probability * self.emission_probability[
                        '{}|{}'.format(self.word_set[word], tag)]
                self.backtrack[tag].append(backtrack)

        # Calculates Max Probability Path
        max_probability = 0
        backtrack_index = self.tag_set[0]
        for tag in range(1, len(self.tag_set)):
            current_tag_probability = float(self.viterbi[self.tag_set[tag]][len(self.word_set) - 1]) * \
                                      float(self.transition_probability['{}|{}'.format('.', self.tag_set[tag])])

            if current_tag_probability > max_probability:
                max_probability = current_tag_probability
                backtrack_index = self.tag_set[tag]

        # Uses backtracking to find states.
        self.states.insert(0, backtrack_index)
        for word in range(len(self.word_set) - 2, -1, -1):
            self.states.insert(0, self.backtrack[backtrack_index][word])
            backtrack_index = self.backtrack[backtrack_index][word]

        return self.states
