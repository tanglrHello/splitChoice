file_splitter = "!@#"
out_file_splitter = ","

csv_col_names = ['number',
                 'text',
                 'splitinfo',
                 'segres',
                 'segres_fg',
                 'auto_seg',
                 'auto_seg_fg',
                 'posres',
                 'auto_pos',
                 'goldtimes',
                 'auto_time',
                 'goldlocs',
                 'goldterms',
                 'auto_loc',
                 'goldquants',
                 'bpres',
                 'auto_bpres',
                 'topTemplate',
                 'topTemplateTypes',
                 'topTemplateCueword',
                 'secondTemplate',
                 'secondTemplateTypes',
                 'secondTemplateCueword',
                 'choiceQuestionSentence',
                 'auto_topTemplate',
                 'auto_secondTemplate',
                 'auto_topTemplateCueword',
                 'auto_choiceQuestionSentence',
                 'auto_secondTemplateCueword',
                 'auto_topTemplateTypes',
                 'auto_secondTemplateTypes',
                 'choice_type',
                 'qiandao_type',
                 'core_type',
                 'core_verb',
                 'delete_part',
                 'context']

index_dict = {}
for col_index, name in enumerate(csv_col_names):
    index_dict[name] = col_index + 1
index_dict['source'] = 0


def print_function_info(func_name):
    print "**************** " + func_name + " ****************"
