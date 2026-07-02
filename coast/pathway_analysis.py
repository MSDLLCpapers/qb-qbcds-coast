import gseapy
from gseapy.plot import barplot, dotplot
from gprofiler import GProfiler
import matplotlib.pyplot as plt
import seaborn as sns

def gseapy_analysis(gene_names, organism='Human', gene_sets='GO_Biological_Process_2018', outdir='gsea_top_hic'):
    enr = gseapy.enrichr(gene_list=gene_names, gene_sets=gene_sets, organism=organism,
                         outdir=outdir,format='png',
                         verbose=False)
    return enr

def gprofiler_analysis(gene_names, organism='hsapiens'):
    gp = GProfiler(return_dataframe=True)
    results = gp.profile(organism=organism, query=gene_names)
    return results