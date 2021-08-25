from readset import dataset

def test_stats(name, sample_cut=0.3):
    data = dataset(name)
    data.load_anatations()
    data.error_calulation(sample_cut)
    data.error_t_test()
    # data.direction_selftest()
    # data.label_selftest()


    # data.save_df('error_df.latex', data.error_df)
    # data.save_df('direction_degdf_df.latex', data.direction_degdf_df)
    # data.save_df('ftest_pos_df.latex', data.ftest_pos_df)
    # data.save_df('human_error_df.latex', data.human_error_df)
    # data.save_df('openp_error_df.latex', data.openp_error_df)
    # data.save_df('error_degdf_df.latex', data.error_degdf_df)
    #data.save_df('.latex', data.)
    #data.save_df('.latex', data.)
    #data.save_df('.latex', data.)
    #data.save_df('.latex', data.)
    #data.save_df('.latex', data.)



if __name__ == '__main__':
    test_stats("P2")
