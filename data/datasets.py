import pandas as pd
import re
import numpy as np

class OTEDataFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return OTEDataFrame

    @staticmethod
    def load_data(report_dir, **kwargs):
        if report_dir.endswith('.txt'):
            data = pd.read_csv(report_dir, **kwargs)
        elif report_dir.endswith('.csv'):
            data = pd.read_csv(report_dir, **kwargs)
        elif report_dir.endswith('.xlsx'):
            data = pd.read_excel(report_dir, **kwargs)
        else:
            try:
                data = pd.read_pickle(report_dir, **kwargs)
            except:
                raise ValueError("Unknown file type. Please check.")

        return OTEDataFrame(data)

    def filter_date(self, start_date, end_date, date_column, **kwargs):
        #convert date_column to datetime
        self[date_column] = pd.to_datetime(self[date_column])
        #convert start and end dates to date_times
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        #filter date
        indexes = (self[date_column] >= start_date) & (self[date_column] < end_date)

        df = self.loc[indexes]

        return df.reset_index(**kwargs)

    def filter_test(self, test, test_column, **kwargs):
        # validate test argument
        if not any([test.lower() == i for i in ('ote', 'otefs')]):
            raise ValueError("'test' argument must be either 'ote' or 'otefs'.")

        test_filter = self[test_column].str.contains('for Schools')

        if test == 'ote':
            test_filter = -test_filter

        return self[test_filter].reset_index(**kwargs)

    def remove_data(self, to_remove):
        for key in to_remove:
            column_name = key
            entries_to_remove = to_remove[key]

            to_remove_indexes = self[column_name].astype(str).isin(entries_to_remove)
            df = self.loc[~to_remove_indexes]

        return df



    def make_runfile(self, item_column, response_column, key_column,
                     instance_columns, save_dir,
                     template_dir, analysis_name, anchors=None, item_diff_column=None):
        # create winsteps runfile from data in current state


        # validate anchor arguments
        if anchors is not None and item_diff_column:
            raise ValueError("To extract anchor difficulties 'item_diff_column' must be supplied.")
        elif anchors and item_diff_column is not None:
            raise ValueError("'item_diff_column' is only required when anchor items are supplied. ")
        print('Anchors validated')


        tt_instances = self[instance_columns].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        print('tt_instances created.')
        # extract only first element in each keys string. this is to avoid complications with 'A-B' key format in key column
        self[key_column] = [key[0] for key in self[key_column]]
        self[response_column] = [key[0] for key in self[response_column]]

        # create df of all items
        item_df = self.drop_duplicates(subset=item_column)

        # if anchors are present item difficulty is included in item_df for later extraction
        if item_diff_column:
            item_df_cols = [item_column, item_diff_column, key_column]
        else:
            item_df_cols = [item_column, key_column]
        item_df = item_df[item_df_cols]

        # create index dictionaries for items, tts, and keys
        print('Indexing items and test takers...')
        report_items_dict = {item_id: index for index, item_id in enumerate(sorted(set(self[item_column])))}
        print('report_items_dict created.')
        report_tts_dict = {tt_id: index for index, tt_id in enumerate(sorted(set(tt_instances)))}
        print('report_tts_dict created.')
        key_dict = {item: item_df.loc[item_df[item_column]==item, key_column].values.item() for item in report_items_dict.keys()}
        print('report_items_dict created.')

        n_tts = len(report_tts_dict)
        # number of items (NI)
        n_items = len(report_items_dict)

        ## create response matrix
        print('Creating response matrix...')
        data_mat = np.full([n_tts, n_items], 'N')
        for tt, item, response in zip(tt_instances, self[item_column], self[response_column]):
            tt_idx = report_tts_dict[tt]
            item_idx = report_items_dict[item]
            data_mat[tt_idx, item_idx] = response
        print('Response matrix created')
        # convert matrix to correct string format for output funfile
        data_mat_joined = np.array(["".join(item) for item in data_mat])
        print('Matrix converted to string')

        responses_df = pd.DataFrame({
            'tt_id': list(report_tts_dict.keys()),
            'responses': data_mat_joined
        })

        responses_string = responses_df.to_string(index=False, header=False, justify='initial')
        items_string = '\n'.join(report_items_dict.keys())
        print('Response matrix created.')

        ## extract anchors
        if anchors:
            print('Extracting, matching and indexing anchor items...')
            anchors = pd.DataFrame({item_column: anchors})
            anchor_diffs_df = anchors.merge(self, on=item_column, how='left')
            anchor_items_indexes = [str(report_items_dict[anchor] + 1) for anchor in anchors]

            # get anchor items and convert to string with indexes for output runfile
            n_anchors = len(anchors)
            anchor_items_string = ['\t'] * (n_anchors * 4)
            anchor_items_string[::4] = anchor_items_indexes
            anchor_items_string[2::4] = anchor_diffs_df.astype(str)
            anchor_items_string[3::4] = ['\n'] * n_anchors
            anchor_items_string = ''.join(anchor_items_string)

            print(anchors)



        print('Constructing runfile...')
        # get index of first response (ITEM1)
        unique_keys = self[key_column].unique()
        unique_keys_str = ''.join(unique_keys)
        #print('Unique keys created.')

        first_item = re.search(r'\s[{}N]'.format(unique_keys_str), responses_string).end()
        #print('First items located.')
        # get length of tt_id (NAMELEN)
        id_len = max([len(id) for id in report_tts_dict.keys()])
        #print('ID length extracted.')
        # get codes (CODES)
        #print(unique_keys_str)
        codes = unique_keys_str
        #print('Codes extracted: '+codes)
        # create string for (KEYS)
        #print(key_dict)
        keys_string = ''.join(list(key_dict.values()))

        ###### add data to template
        ##functions for adding data to runfile
        def insert_variable(variable, insert_val, string, needs_idx_adj=False):
            if needs_idx_adj:
                insert_val = insert_val + 1

            insert_val = str(insert_val)
            reg_find = variable + r'\s*=\s*<VALUE>\s*\n'
            replace_reg = variable + ' = ' + insert_val + '\n'
            replaced_string = re.sub(reg_find, replace_reg, string)

            return replaced_string

        print('Writing and saving runfile...')
        rf = open(template_dir, 'r')
        rf_string = rf.read()
        rf.close()

        rf_string = insert_variable('TITLE', analysis_name, rf_string)
        #print('title inserted')
        rf_string = insert_variable('ITEM1', first_item, rf_string)
        #print('item1 inserted')
        rf_string = insert_variable('NI', n_items, rf_string)
        #print('n items inserted')
        rf_string = insert_variable('NAMELEN', id_len, rf_string)
        #print('name len inserted')
        rf_string = insert_variable('CODES', codes, rf_string)
        #print('codes inserted')
        rf_string = insert_variable('KEY', keys_string, rf_string)
        if anchors:
            rf_string = rf_string.replace('<ANCHORFILE>', anchor_items_string)

        rf_string = rf_string.replace('<ITEMIDS>', items_string)
        #print('item ids inserted')
        rf_string = rf_string.replace('<RESPONSEDATA>', responses_string)
        #print('response data inserted')

        rf_new = open('{}.txt'.format(save_dir), 'w')
        rf_new.write(rf_string)
        rf_new.close()
        print("Runfile created and saved for {} at: {}.".format(analysis_name, save_dir))

        ### print summary information
        print('{} SUMAMRY INFORMATION {}'.format('*'*10, '*'*10))
        print("n test takers = {}".format(n_tts))
        print("n items = {}".format(n_items))


