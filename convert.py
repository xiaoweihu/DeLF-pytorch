from qd.tsv_io import tsv_reader, tsv_writer
from qd.qd_common import json_dump

def convert_matching_format(infile, outfile):
    def gen_rows():
        for parts in tsv_reader(infile):
            assert(len(parts) == 3)
            pred_labels = [it.split(':') for it in parts[2].split(';')]
            pred_labels = [[label, int(score), ""] for label, score in pred_labels]
            yield parts[0], parts[1], json_dump(pred_labels)
    tsv_writer(gen_rows(), outfile)

if __name__ == "__main__":
    infile = "output/delf_brandsports_bneval/logo40_delf_query.tsv"
    outfile = "output/delf_brandsports_bneval/delf_query_logo40.test.2.0.query.delf_to_logo40can2.train.2.0.index.delf.tsv"
    convert_matching_format(infile, outfile)
    from run import calculate_accuracy_for_matching
    calculate_accuracy_for_matching(outfile, (1, 5))
