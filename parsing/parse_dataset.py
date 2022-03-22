from Drain import *

# def parse_logs(input_dir, log_data_frame, output_dir, log_format, rex=[], st=0.15, depth=2, log_file='tbird.csv'):
def parse_logs(input_dir, log_data_frame, output_dir, log_format, rex=[], st=0.5, depth=4, log_file='original_HDFS.log'):
    regex = rex  # empty, we don't have to use it.
    parser = LogParser(log_format, log_data_frame, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex)
    templates, df_templates = parser.parse(log_file)
    print("Finished parsing logs!!!")
    return templates, df_templates

path = "../data/hdfs"
data = pd.read_csv(path+"/payload_hdfs.csv", sep=",")
a, b = parse_logs(input_dir=path,  log_data_frame=data, output_dir="./results/", log_format="<Payload><Label>")
a.to_csv("../data/hdfs/hdfs_structured.csv", index=False)
b.to_csv("../data/hdfs/hdfs_templates.csv", index=False)
