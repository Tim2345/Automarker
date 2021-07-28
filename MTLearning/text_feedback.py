import numpy as np
import html



class MultitaskFeedback(object):


    def get_weights(self, binary_preds_out, how):
        if how == 'incorrect' or how == 'correct':
            col_index = None
            if how == 'incorrect':
                col_index = 0

            elif how == 'correct':
                col_index = 1

            if len(binary_preds_out) > 1:
                weights = []
                for arr in binary_preds_out:
                    weights.append(arr[:, col_index])
            else:
                weights = binary_preds_out[:, col_index]

        elif callable(how):
            if len(binary_preds_out) > 1:
                weights = [how(binary_preds) for binary_preds in binary_preds_out]
            else:
                weights = how(binary_preds_out)

        return weights


    def html_escape(self, text):
        return html.escape(text)

    def highlight_text(self, tokens, weights, extras=None, max_alpha=1, adjustment=0):
        '''

        :param text: list of tokens of same length as values
        :param weights: weightings to be used for highlighting
        :param extras: extra information about each script to be pasted above the text on the html printout
        :return:
        '''

        highlighted_text = [extras + ' <Br>']

        for token, weight in zip(tokens, weights):

            if weight is not None:
                val = (weight / max_alpha) + adjustment
                highlighted_text.append(
                    '<span style="background-color:rgba(135,206,250,' +
                    str(val) +
                    ');">' +
                    self.html_escape(token) +
                    '</span>'
                )

            else:

                highlighted_text.append(token)

        highlighted_text = ' '.join(highlighted_text)

        return highlighted_text


    def highlight_texts(self, tokens_list, weights_list, extras_list, max_alpha=1, adjustment=0):

        html_strings = []

        for text, weight, extra in zip(tokens_list, weights_list, extras_list):
            highlighted_string = self.highlight_text(
                tokens=text,
                weights=weight,
                extras=extra,
                max_alpha=max_alpha,
                adjustment=adjustment
            )

            html_strings.append(highlighted_string)

        return '<P>'.join(html_strings)



    def to_file(self, html_string):
        f = open("text_output_cola.html", "w")
        f.write(html_string)
        f.close()

