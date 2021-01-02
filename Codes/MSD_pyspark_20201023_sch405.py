# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------



# Python and pyspark modules required

import sys
import numpy as np

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.sql import functions as F
from pyspark.sql.window import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.ml import Pipeline


# Required to allow the file to be submitted and run using spark-submit instead
# of using pyspark interactively

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()

# Compute suitable number of partitions

conf = sc.getConf()

N = int(conf.get("spark.executor.instances"))
M = int(conf.get("spark.executor.cores"))
partitions = 4 * N * M     # 32


# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------

# Define the data frame schema
mismatches_schema = StructType([
    StructField("song_id", StringType(), True),    
    StructField("song_artist", StringType(), True),
    StructField("song_title", StringType(), True),
    StructField("track_id", StringType(), True),
    StructField("track_artist", StringType(), True),
    StructField("track_title", StringType(), True)
])

# Load the data from local file to master node python (not spark), and doing the data processing directly. 
# The dataset uploading should be small, because this process is happen in master node memory.
with open("/scratch-network/courses/2020/DATA420-20S2/data/msd/tasteprofile/mismatches/sid_matches_manually_accepted.txt", "r") as f:   #open the local file read as f
    lines = f.readlines()                              # assign object "lines" as a list which contains each line in the 
    sid_matches_manually_accepted = []                 # A new list which used to store the information of each line
    for line in lines:                                 # Iterate the line string process (split the feature string and comebine them with tuple structure) to each line 
        if line.startswith("< ERROR: "):               # Find the target information pattern and skip the useless information
            a = line[10:28]                            # Using index to stract the "song_id" information as a string.
            b = line[29:47]                            # Using index to stract the "track_id" information as a string.
            c, d = line[49:-1].split("  !=  "  )       # Using '!=' to split the residual string into two parts.
            e, f = c.split("  -  ")                    # Using '-' to split c into two parts "song_artist" and "song_title".
            g, h = d.split("  -  ")                    # Using '-' to split d into two parts "track_artist" and "track_title".
            sid_matches_manually_accepted.append((a, e, f, b, g, h))   # append the tuple of each line into the new list during the for loop.

matches_manually_accepted = spark.createDataFrame(sc.parallelize(sid_matches_manually_accepted, 8), schema=mismatches_schema)  
                                                       # upload and parallelize the result list into spark as dataframe with the defined schema 
matches_manually_accepted.cache()                      # cache the dataframe    
print(matches_manually_accepted.count())               # 488   Why it is not 938?  
matches_manually_accepted.show(10, 40)


# +------------------+-----------------+----------------------------------------+------------------+----------------------------------------+----------------------------------------+
# |           song_id|      song_artist|                              song_title|          track_id|                            track_artist|                             track_title|
# +------------------+-----------------+----------------------------------------+------------------+----------------------------------------+----------------------------------------+
# |SOFQHZM12A8C142342|     Josipa Lisac|                                 razloga|TRMWMFG128F92FFEF2|                            Lisac Josipa|                            1000 razloga|
# |SODXUTF12AB018A3DA|       Lutan Fyah|     Nuh Matter the Crisis Feat. Midnite|TRMWPCD12903CCE5ED|                                 Midnite|                   Nah Matter the Crisis|
# |SOASCRF12A8C1372E6|Gaetano Donizetti|L'Elisir d'Amore: Act Two: Come sen v...|TRMHIPJ128F426A2E2|Gianandrea Gavazzeni_ Orchestra E Cor...|L'Elisir D'Amore_ Act 2: Come Sen Va ...|
# |SOITDUN12A58A7AACA|     C.J. Chenier|                               Ay, Ai Ai|TRMHXGK128F42446AB|                         Clifton Chenier|                               Ay_ Ai Ai|
# |SOLZXUM12AB018BE39|           許志安|                                男人最痛|TRMRSOF12903CCF516|                                Andy Hui|                        Nan Ren Zui Tong|
# |SOTJTDT12A8C13A8A6|                S|                                       h|TRMNKQE128F427C4D8|                             Sammy Hagar|                 20th Century Man (Live)|
# |SOGCVWB12AB0184CE2|                H|                                       Y|TRMUNCZ128F932A95D|                                Hawkwind|                25 Years (Alternate Mix)|
# |SOKDKGD12AB0185E9C|     影山ヒロノブ|Cha-La Head-Cha-La (2005 ver./DRAGON ...|TRMOOAH12903CB4B29|                        Takahashi Hiroki|Maka fushigi adventure! (2005 Version...|
# |SOPPBXP12A8C141194|    Αντώνης Ρέμος|                        O Trellos - Live|TRMXJDS128F42AE7CF|                           Antonis Remos|                               O Trellos|
# |SODQSLR12A8C133A01|    John Williams|Concerto No. 1 for Guitar and String ...|TRWHMXN128F426E03C|               English Chamber Orchestra|II. Andantino siciliano from Concerto...|


with open("/scratch-network/courses/2020/DATA420-20S2/data/msd/tasteprofile/mismatches/sid_mismatches.txt", "r") as f:
    lines = f.readlines()
    sid_mismatches = []
    for line in lines:
        if line.startswith("ERROR: "):
            a = line[8:26]
            b = line[27:45]
            c, d = line[47:-1].split("  !=  ")
            e, f = c.split("  -  ")
            g, h = d.split("  -  ")
            sid_mismatches.append((a, e, f, b, g, h))

mismatches = spark.createDataFrame(sc.parallelize(sid_mismatches, 64), schema=mismatches_schema)
mismatches.cache()
print(mismatches.count())  # 19094
mismatches.show(10, 40)


# +------------------+-------------------+----------------------------------------+------------------+--------------+----------------------------------------+
# |           song_id|        song_artist|                              song_title|          track_id|  track_artist|                             track_title|
# +------------------+-------------------+----------------------------------------+------------------+--------------+----------------------------------------+
# |SOUMNSI12AB0182807|Digital Underground|                        The Way We Swing|TRMMGKQ128F9325E10|      Linkwood|           Whats up with the Underground|
# |SOCMRBE12AB018C546|         Jimmy Reed|The Sun Is Shining (Digitally Remaste...|TRMMREB12903CEB1B1|    Slim Harpo|               I Got Love If You Want It|
# |SOLPHZY12AC468ABA8|      Africa HiTech|                                Footstep|TRMMBOC12903CEB46E|Marcus Worgull|                 Drumstern (BONUS TRACK)|
# |SONGHTM12A8C1374EF|     Death in Vegas|                            Anita Berber|TRMMITP128F425D8D0|     Valen Hsu|                                  Shi Yi|
# |SONGXCA12A8C13E82E| Grupo Exterminador|                           El Triunfador|TRMMAYZ128F429ECE6|     I Ribelli|                               Lei M'Ama|
# |SOMBCRC12A67ADA435|      Fading Friend|                             Get us out!|TRMMNVU128EF343EED|     Masterboy|                      Feel The Heat 2000|
# |SOTDWDK12A8C13617B|       Daevid Allen|                              Past Lives|TRMMNCZ128F426FF0E| Bhimsen Joshi|            Raga - Shuddha Sarang_ Aalap|
# |SOEBURP12AB018C2FB|  Cristian Paduraru|                              Born Again|TRMMPBS12903CE90E1|     Yespiring|                          Journey Stages|
# |SOSRJHS12A6D4FDAA3|         Jeff Mills|                      Basic Human Design|TRMWMEL128F421DA68|           M&T|                           Drumsettester|
# |SOIYAAQ12A6D4F954A|           Excepter|                                      OG|TRMWHRI128F147EA8E|    The Fevers|Não Tenho Nada (Natchs Scheint Die So...|
# +------------------+-------------------+----------------------------------------+------------------+--------------+----------------------------------------+


triplets_schema = StructType([
    StructField("user_id", StringType(), True),
    StructField("song_id", StringType(), True),
    StructField("plays", IntegerType(), True)
])

# The triplets file is large, so load to spark from hdfs directly.
triplets = (
    spark.read.format("csv")
    .option("header", "false")
    .option("delimiter", "\t")
    .option("codec", "gzip")
    .schema(triplets_schema)
    .load("hdfs:///data/msd/tasteprofile/triplets.tsv/")
    .cache()
)
triplets.cache()
triplets.count()     # 48373586
triplets.show(10, 50)

# +----------------------------------------+------------------+-----+
# |                                 user_id|           song_id|plays|
# +----------------------------------------+------------------+-----+
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOQEFDN12AB017C52B|    1|
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOQOIUJ12A6701DAA7|    2|
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOQOKKD12A6701F92E|    4|
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOSDVHO12AB01882C7|    1|
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOSKICX12A6701F932|    1|
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOSNUPV12A8C13939B|    1|
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOSVMII12A6701F92D|    1|
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOTUNHI12B0B80AFE2|    1|
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOTXLTZ12AB017C535|    1|
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOTZDDX12A6701F935|    1|
# +----------------------------------------+------------------+-----+



# Remove the records which should not be put into mismatched by using left_anti join by the key "song_id" 
# Drop only one record...why
mismatches_not_accepted = mismatches.join(matches_manually_accepted, on="song_id", how="left_anti")

# Remove the mismatched records from triplets by using left_anti join by the key "song_id" 
# Drop 2,578,475 mismatched records
triplets_not_mismatched = triplets.join(mismatches_not_accepted, on="song_id", how="left_anti")

# Repartition the result to make it more balance to computate
triplets_not_mismatched = triplets_not_mismatched.repartition(partitions).cache()

print(mismatches_not_accepted.count())  # 19093
print(triplets.count())                 # 48373586  
print(triplets_not_mismatched.count())  # 45795111

# Processing Q2 (b)

# The command below is checking the unique attributes schema in the attributes file in hdfs
# hdfs dfs -cat "/data/msd/audio/attributes/*" | awk -F',' '{print $2}' | sort | uniq

# NUMERIC
# real
# real 
# string
# string
# STRING

# Map the attribute type into formal schema data type
audio_attribute_type_mapping = {
  "NUMERIC": DoubleType(),
  "real": DoubleType(),
  "string": StringType(),
  "STRING": StringType()
}

# Create a list store all the audio dataset names in the audio/attributes
audio_dataset_names = [
  "msd-jmir-area-of-moments-all-v1.0",
  "msd-jmir-lpc-all-v1.0",
  "msd-jmir-methods-of-moments-all-v1.0",
  "msd-jmir-mfcc-all-v1.0",
  "msd-jmir-spectral-all-all-v1.0",
  "msd-jmir-spectral-derivatives-all-all-v1.0",
  "msd-marsyas-timbral-v1.0",
  "msd-mvd-v1.0",
  "msd-rh-v1.0",
  "msd-rp-v1.0",
  "msd-ssd-v1.0",
  "msd-trh-v1.0",
  "msd-tssd-v1.0"
]

audio_dataset_schemas = {}   # Creat a new dictionary
for audio_dataset_name in audio_dataset_names:
  print(audio_dataset_name)

  audio_dataset_path = f"/scratch-network/courses/2020/DATA420-20S2/data/msd/audio/attributes/{audio_dataset_name}.attributes.csv"
  with open(audio_dataset_path, "r") as f:
    rows = [line.strip().split(",") for line in f.readlines()]
    # string strip() function will remove spaces at the beginning and at the end of the string:

  # you could rename feature columns with a short generic name

    rows[-1][0] = "track_id"
    for i, row in enumerate(rows[0:-1]):
        row[0] = f"feature_{i:04d}"
  
# Define the schema of audio_dataset
  audio_dataset_schemas[audio_dataset_name] = StructType([
    StructField(row[0], audio_attribute_type_mapping[row[1]], True) for row in rows
  ])
    
  s = str(audio_dataset_schemas[audio_dataset_name])
  print(s[0:50] + " ... " + s[-50:])


# msd-jmir-area-of-moments-all-v1.0
# StructType(List(StructField(Area_Method_of_Moments ... e,true),StructField(MSD_TRACKID,StringType,true)))
# msd-jmir-lpc-all-v1.0
# StructType(List(StructField(LPC_Overall_Standard_D ... e,true),StructField(MSD_TRACKID,StringType,true)))
# msd-jmir-methods-of-moments-all-v1.0
# StructType(List(StructField(Method_of_Moments_Over ... e,true),StructField(MSD_TRACKID,StringType,true)))
# msd-jmir-mfcc-all-v1.0
# StructType(List(StructField(MFCC_Overall_Standard_ ... e,true),StructField(MSD_TRACKID,StringType,true)))
# msd-jmir-spectral-all-all-v1.0
# StructType(List(StructField(Spectral_Centroid_Over ... e,true),StructField(MSD_TRACKID,StringType,true)))
# msd-jmir-spectral-derivatives-all-all-v1.0
# StructType(List(StructField(Spectral_Centroid_Over ... e,true),StructField(MSD_TRACKID,StringType,true)))
# msd-marsyas-timbral-v1.0
# StructType(List(StructField(Mean_Acc5_Mean_Mem20_Z ... Type,true),StructField(track_id,StringType,true)))
# msd-mvd-v1.0
# StructType(List(StructField("component_0",DoubleTy ... ,true),StructField(instanceName,StringType,true)))
# msd-rh-v1.0
# StructType(List(StructField("component_0",DoubleTy ... ,true),StructField(instanceName,StringType,true)))
# msd-rp-v1.0
# StructType(List(StructField("component_1",DoubleTy ... ,true),StructField(instanceName,StringType,true)))
# msd-ssd-v1.0
# StructType(List(StructField("component_0",DoubleTy ... ,true),StructField(instanceName,StringType,true)))
# msd-trh-v1.0
# StructType(List(StructField("component_0",DoubleTy ... ,true),StructField(instanceName,StringType,true)))
# msd-tssd-v1.0
# StructType(List(StructField("component_1",DoubleTy ... ,true),StructField(instanceName,StringType,true)))




# Audio Similarity Q1 (a)

# you can now use these schemas to load any one of the above datasets
# -----------------------------------------------------------------------------
# Data analysis
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Audio features descriptive statistics and correlated
# -----------------------------------------------------------------------------
# 
# Python and pyspark modules required


# Restart: start_pyspark_shell -e 4 -c 2 -w 4 -m 8
# It will clear the memory
import sys
import numpy as np
import pandas as pd

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.sql import functions as F
from pyspark.sql.window import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.ml import Pipeline


# Required to allow the file to be submitted and run using spark-submit instead
# of using pyspark interactively

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()

# Compute suitable number of partitions

conf = sc.getConf()

N = int(conf.get("spark.executor.instances"))
M = int(conf.get("spark.executor.cores"))
partitions = 4 * N * M     # 32


# Map the attribute type into formal schema data type
audio_attribute_type_mapping = {
  "NUMERIC": DoubleType(),
  "real": DoubleType(),
  "string": StringType(),
  "STRING": StringType()
}

# Create a list store all the audio dataset names in the audio/attributes
audio_dataset_names = [
  "msd-jmir-area-of-moments-all-v1.0",
  "msd-jmir-lpc-all-v1.0",
  "msd-jmir-methods-of-moments-all-v1.0",
  "msd-jmir-mfcc-all-v1.0",
  "msd-jmir-spectral-all-all-v1.0",
  "msd-jmir-spectral-derivatives-all-all-v1.0",
  "msd-marsyas-timbral-v1.0",
  "msd-mvd-v1.0",
  "msd-rh-v1.0",
  "msd-rp-v1.0",
  "msd-ssd-v1.0",
  "msd-trh-v1.0",
  "msd-tssd-v1.0"
]

audio_dataset_schemas = {}   # Creat a new dictionary
for audio_dataset_name in audio_dataset_names:
#  print(audio_dataset_name)

  audio_dataset_path = f"/scratch-network/courses/2020/DATA420-20S2/data/msd/audio/attributes/{audio_dataset_name}.attributes.csv"
  with open(audio_dataset_path, "r") as f:
    rows = [line.strip().split(",") for line in f.readlines()]
    # string strip() function will remove spaces at the beginning and at the end of the string:

  # Rename feature columns with a short generic name

    rows[-1][0] = "track_id"
    for i, row in enumerate(rows[0:-1]):
        row[0] = f"f_{i:04d}" # "f" means "feature"
  
# Define the schema of audio_dataset
  audio_dataset_schemas[audio_dataset_name] = StructType([
    StructField(row[0], audio_attribute_type_mapping[row[1]], True) for row in rows
  ])
    
  s = str(audio_dataset_schemas[audio_dataset_name])
  
  
# Define the feature dataset schema
feature_dataset_v1_schema = audio_dataset_schemas["msd-jmir-spectral-all-all-v1.0"]

# Load the feature dataset I chose with the defined schema
feature_dataset_v1 = (
    spark.read.format("csv")
    .option("inferSchema", "False")
    .option("header", "true")
    .schema(feature_dataset_v1_schema)
    .load("/data/msd/audio/features/msd-jmir-spectral-all-all-v1.0.csv/")
    .repartition(partitions)
)
'''
feature_v1_subset = (
    feature_dataset_v1.
    select(F.col("track_id"),F.col("f_1"),F.col("f_0000"),F.col("f_0089"),F.col("f_0155"),F.col("f_0220"),F.col("f_0315"),F.col("f_0419"))
)
'''
feature_dataset_v1.cache()
feature_dataset_v1.count()  # pyspark 994615  # hdfs 994180 lines

feature_dataset_v1.printSchema()

# root
 # |-- f_0000: double (nullable = true)
 # |-- f_0001: double (nullable = true)
 # |-- f_0002: double (nullable = true)
 # |-- f_0003: double (nullable = true)
 # |-- f_0004: double (nullable = true)
 # |-- f_0005: double (nullable = true)
 # |-- f_0006: double (nullable = true)
 # |-- f_0007: double (nullable = true)
 # |-- f_0008: double (nullable = true)
 # |-- f_0009: double (nullable = true)
 # |-- f_0010: double (nullable = true)
 # |-- f_0011: double (nullable = true)
 # |-- f_0012: double (nullable = true)
 # |-- f_0013: double (nullable = true)
 # |-- f_0014: double (nullable = true)
 # |-- f_0015: double (nullable = true)
 # |-- track_id: string (nullable = true)



audio_features_statistics = (
    feature_dataset_v1
    .select([col for col in feature_dataset_v1.columns if col.startswith("f")])
    .describe()
    .toPandas()
    .set_index("summary")
    .rename_axis(None)
    .T
)
print(audio_features_statistics)


         # count                   mean                 stddev  min      max
# f_0000  994615      6.945075321958743     3.6318040939391563  0.0    73.31
# f_0001  994615    0.05570656330077471   0.026500210060303352  0.0   0.3739
# f_0002  994615  0.0039454294223152265    0.00326533334345565  0.0  0.07164
# f_0003  994615      222.5178576955708     59.726390795145896  0.0  10290.0
# f_0004  994615  0.0022271413534831882   0.001039740410980691  0.0  0.01256
# f_0005  994615    0.07420121987116546    0.03176619445967485  0.0   0.3676
# f_0006  994615   0.060207318112033295    0.01851647926605086  0.0   0.4938
# f_0007  994615      16.80284722623327      7.530133216657491  0.0    141.6
# f_0008  994615      9.110257231921896     3.8436388429686748  0.0    133.0
# f_0009  994615   0.061943205894793366    0.02901672982497254  0.0   0.7367
# f_0010  994615  0.0029321551802922404   0.002491152580972017  0.0  0.07549
# f_0011  994615     1638.7321744558617     106.10634441394467  0.0  24760.0
# f_0012  994615   0.004395476870406143  0.0019958918239917425  0.0  0.02366
# f_0013  994615    0.16592784214140108    0.07429866377222316  0.0   0.8564
# f_0014  994615     0.5562831490668243     0.0475538009894805  0.0   0.9538
# f_0015  994615      26.68041063976806     10.394734197335117  0.0    280.5


# Correlations

# feature_dataset_v1_cor = feature_dataset_v1.drop("track_id")
# col_name = feature_dataset_v1_cor.columns
# features = feature_dataset_v1_cor.rdd.map(lambda row: row[0:])
# corr_mat = audio_features_statistics.corr(features, method = "pearson")
# corr_df = pd.DataFrame(corr_mat)
# corr_df.index, corr_df.columns = col_names, col_names

inputCols = [col for col in feature_dataset_v1.columns if col.startswith("f")]   # only select the feature columns, ignore the "track_id"
assembler = VectorAssembler(
    inputCols=inputCols,
    outputCol="Features"       # convert each feature column(variable values) into vector
).setHandleInvalid("skip")     # skip the the line which has missing values

features = assembler.transform(feature_dataset_v1).select(["Features"])
features.cache()
# features.count() Error
features.show(10, 100)

# +----------------------------------------------------------------------------------------------------+
# |                                                                                            Features|
# +----------------------------------------------------------------------------------------------------+
# |[10.14,0.07024,0.006577,233.7,0.003728,0.1198,0.04453,22.63,9.88,0.06322,0.004728,1592.0,0.00583,...|
# |[7.652,0.06613,0.009398,192.3,0.002984,0.1063,0.05237,19.82,11.54,0.07887,0.006133,1568.0,0.00657...|
# |[3.376,0.02469,1.33E-4,206.2,6.315E-4,0.02449,0.05468,8.115,9.793,0.06264,7.183E-5,1697.0,8.488E-...|
# |[6.897,0.06022,0.006882,208.6,0.004006,0.13,0.07045,19.78,5.726,0.03521,0.004171,1606.0,0.004992,...|
# |[8.129,0.06202,0.004096,264.6,0.002456,0.08884,0.1061,18.9,11.1,0.07513,0.00219,1647.0,0.004249,0...|
# |[4.196,0.03996,0.001413,227.4,0.001818,0.06656,0.0577,12.55,7.966,0.04703,0.001323,1689.0,0.00400...|
# |[5.877,0.05685,0.004309,186.5,0.002162,0.07004,0.04124,14.67,10.6,0.07633,0.003688,1629.0,0.00494...|
# |[10.62,0.06326,0.005228,281.5,0.002999,0.09503,0.07862,21.71,6.896,0.04154,0.002159,1594.0,0.0040...|
# |[5.304,0.05405,0.00182,206.2,0.001402,0.05046,0.0575,14.64,9.054,0.06126,8.559E-4,1700.0,0.002536...|
# |[5.586,0.0523,0.002224,237.4,0.001623,0.05353,0.0438,15.06,6.49,0.0472,0.002027,1573.0,0.003734,0...|
# +----------------------------------------------------------------------------------------------------+
# only showing top 10 rows


# Correlation matrix. Syntex: Correlation.corr(features, {col}, {method}).collect()[0][0].toArray()
correlations = Correlation.corr(features, 'Features', 'pearson').collect()[0][0].toArray()
# Define the threshold to select strong correlated features
threshold = 0.7
# Imagine the correlation matrix, the values of the diagonal (autocorrelation) are "1", 
# the number of "1" is the number of lines (columns).
# the number in the correlation matrix are 
num_correlated_columns_not_the_same = (correlations > threshold).sum() - correlations.shape[0]

print((correlations > threshold).astype(int))

# [[1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
 # [1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
 # [0 0 1 0 1 1 0 0 0 0 1 0 1 1 0 0]
 # [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0]
 # [0 0 1 0 1 1 0 0 0 0 1 0 1 1 0 0]
 # [0 0 1 0 1 1 0 0 0 0 1 0 1 0 0 0]
 # [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0]
 # [1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
 # [0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 1]
 # [0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 1]
 # [0 0 1 0 1 1 0 0 0 0 1 0 1 1 0 0]
 # [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0]
 # [0 0 1 0 1 1 0 0 0 0 1 0 1 1 0 0]
 # [0 0 1 0 1 0 0 0 0 0 1 0 1 1 0 0]
 # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0]
 # [0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 1]]

print(correlations.shape) # (16, 16)
print(num_correlated_columns_not_the_same) # 40

# Check the correlated_pairs
temp = correlations

correlated_pairs = []  # creat blank list used to contain the high correlated_pairs.
correlated_columns = [] 
for i in range(0, temp.shape[0]):   # When some of the features been removed, 
    for j in range(i + 1, temp.shape[1]):  # There are new correlation occure, so need using for loop to handle it.
        if temp[i, j] > 0.7:
            correlated_pairs.append((i, j))
            correlated_columns.append(i)
            correlated_columns.append(j)


correlated_counts = pd.Series(correlated_columns).value_counts().reset_index()
correlated_counts.columns = ["index", "count"]

indexes_to_remove = correlated_counts["index"][correlated_counts["count"] > 1].values
indexes_to_keep = [i for i in range(0, temp.shape[0]) if i not in indexes_to_remove]

temp = temp[indexes_to_keep, :][:, indexes_to_keep]

print(correlated_pairs) 
# [(0, 1), (0, 7), (1, 7), (2, 4), (2, 5), (2, 10), (2, 12), (2, 13), (4, 5), (4, 10), (4, 12), (4, 13), (5, 10), (5, 12), (8, 9), (8, 15), (9, 15), (10, 12), (10, 13), (12, 13)]

print(temp.astype(int))  # The output is different from the output in "Remove one of each pair of correlated variables iteratively" 
                         # This is because this method remove all the corelated_pairs at one time, that is not correct.
                         # It should be done iteratively. First remove the largest one then check then remove next.
 # [[1 0 0 0]
 # [0 1 0 0]
 # [0 0 1 0]
 # [0 0 0 1]]


# Remove one of each pair of correlated variables iteratively 

# value = 2
# counter = 0

n = correlations.shape[0]
counter = 0

indexes = np.array(list(range(0, n)))
matrix = correlations

# while value > 1:
for j in range(0,n):
  mask = matrix > threshold
  sums = mask.sum(axis=0)
  index = np.argmax(sums)
  value = sums[index]
  check = value > 1
#  if value > 1:
  if check:
    k = matrix.shape[0]
    
    keep = [i for i in range(0, k) if i != index]
    matrix = matrix[keep, :][:, keep]
    indexes = indexes[keep]
  else:
    break
  counter += 1
  print(counter) # 1 2 3 4 5 6 7 8






# Check correlations

correlations_new = correlations[indexes, :][:, indexes]
num_correlated_columns_not_the_same = (correlations_new > threshold).sum() - correlations_new.shape

print((correlations_new > threshold).astype(int))
# [[1 0 0 0 0 0 0 0]
 # [0 1 0 0 0 0 0 0]
 # [0 0 1 0 0 0 0 0]
 # [0 0 0 1 0 0 0 0]
 # [0 0 0 0 1 0 0 0]
 # [0 0 0 0 0 1 0 0]
 # [0 0 0 0 0 0 1 0]
 # [0 0 0 0 0 0 0 1]]


print(correlations_new.shape) # (8, 8)

print(num_correlated_columns_not_the_same) # [0 0]



# Assemble vector only from the remaining columns

inputCols = np.array(feature_dataset_v1.columns[:-1])[indexes]
assembler = VectorAssembler(
    inputCols=inputCols,
    outputCol="Features"
).setHandleInvalid("skip")

features = assembler.transform(feature_dataset_v1).select(["Features"])
features.cache()

print(indexes) #[ 3  5  6  7 11 13 14 15]

features.show(10, 100)


# +---------------------------------------------------------+
# |                                                 Features|
# +---------------------------------------------------------+
# |  [233.7,0.1198,0.04453,22.63,1592.0,0.2166,0.5464,26.85]|
# |   [192.3,0.1063,0.05237,19.82,1568.0,0.258,0.5542,33.92]|
# |[206.2,0.02449,0.05468,8.115,1697.0,0.03299,0.5333,24.11]|
# |    [208.6,0.13,0.07045,19.78,1606.0,0.1801,0.5983,20.68]|
# |   [264.6,0.08884,0.1061,18.9,1647.0,0.1648,0.5834,30.68]|
# |  [227.4,0.06656,0.0577,12.55,1689.0,0.1563,0.5413,23.75]|
# |  [186.5,0.07004,0.04124,14.67,1629.0,0.1916,0.584,35.95]|
# | [281.5,0.09503,0.07862,21.71,1594.0,0.1496,0.5624,19.13]|
# | [206.2,0.05046,0.0575,14.64,1700.0,0.09851,0.5466,28.84]|
# |  [237.4,0.05353,0.0438,15.06,1573.0,0.1377,0.5482,21.98]|
# +---------------------------------------------------------+
# only showing top 10 rows


# check this worked by keeping only "indexes"

"""
correlations_new = correlations[indexes, :][:, indexes]
num_correlated_columns_not_the_same = (correlations_new > threshold).sum() - correlations_new.shape[0]
print(num_correlated_columns_not_the_same)
"""

correlations = Correlation.corr(features, 'Features', 'pearson').collect()[0][0].toArray()
num_correlated_columns_not_the_same = (correlations > threshold).sum() - correlations.shape[0]

print((correlations > threshold).astype(int))

# [[1 0 0 0 0 0 0 0]
 # [0 1 0 0 0 0 0 0]
 # [0 0 1 0 0 0 0 0]
 # [0 0 0 1 0 0 0 0]
 # [0 0 0 0 1 0 0 0]
 # [0 0 0 0 0 1 0 0]
 # [0 0 0 0 0 0 1 0]
 # [0 0 0 0 0 0 0 1]]

print(correlations.shape) # (8, 8)

print(num_correlated_columns_not_the_same) # 0

# creat the new feature dataframe without strong correlated features.
feature_col_remian = indexes
print(feature_col_remian) #[ 3  5  6  7 11 13 14 15]

feature_dataset_v1_drop = (
    feature_dataset_v1.select(F.col("f_0002"), 
                              F.col("f_0004"),
                              F.col("f_0005"),
                              F.col("f_0006"),
                              F.col("f_0010"),
                              F.col("f_0012"),
                              F.col("f_0013"),
                              F.col("f_0014"),
                              F.col("track_id")
                            )
    .withColumn("track_id", F.regexp_replace("track_id", "'",""))
)

feature_dataset_v1_drop.show(10, 100)

# +--------+--------+-------+-------+--------+--------+-------+------+------------------+
# |  f_0002|  f_0004| f_0005| f_0006|  f_0010|  f_0012| f_0013|f_0014|          track_id|
# +--------+--------+-------+-------+--------+--------+-------+------+------------------+
# |0.002567|0.002405|0.07963|0.05857|0.001708|0.003733|  0.141|0.5825|TRRESSW12903CFB2D4|
# |0.007234| 0.00355|0.09852|0.04507|0.007577|0.008895| 0.3158| 0.518|TRRUNWZ128F93037FC|
# |0.005577|0.002827|0.09281|0.08254|0.002918|0.004659| 0.1725|0.6845|TRCFZWH128F42375FA|
# |5.648E-4|0.001432|0.05385|0.07864|3.948E-4|0.002002|0.07693|0.5368|TRRXAQZ128F14B0ED2|
# |0.002938|0.002437|0.08972|0.05646|0.002669|0.005186| 0.2016|0.5015|TRBWRWP128EF3546C0|
# |0.001235|0.001064|0.03466|0.04951|0.001284|0.003324| 0.1283| 0.571|TRRMXXA128F92C53F2|
# |0.009883|0.004396| 0.1312|0.04132|0.009299|0.008027| 0.2821|0.4661|TRRIPSR12903CA0285|
# |0.008475|0.004726| 0.1465|0.03453|0.005205|0.006716| 0.2395|0.6277|TRBPPOG12903CEBA04|
# |9.571E-4|0.001914|0.07246|0.09118|9.326E-4|0.004291| 0.1659|0.4653|TRCRUCH128F92FBEC4|
# |4.653E-4|0.001016|0.03638|0.05916| 1.92E-4|0.001392| 0.0524|0.5786|TRCBMKT128F92C96C3|
# +--------+--------+-------+-------+--------+--------+-------+------+------------------+
# only showing top 10 rows




# Audio Similarity Q1 (b)

# -----------------------------------------------------------------------------
# Visualize the distribution of MAGD genres
# -----------------------------------------------------------------------------


# Define the MAGD genre schema
magd_genre_schema = StructType([
    StructField("track_id", StringType(), True),
    StructField("genre_label", StringType(), True)
])

# Load the MAGD genre dataset 

magd_genre_text = (
    spark.read.format("text")
    .load("/data/msd/genre/msd-MAGD-genreAssignment.tsv")
    .repartition(partitions)
)

magd_genre = magd_genre_text.select(
    F.trim(F.substring(F.col('value'), 1, 18)).alias('track_id').cast(StringType()),
    F.trim(F.substring(F.col('value'), 20, 40)).alias('genre').cast(StringType())
)

magd_genre.printSchema()

# root
 # |-- track_id: string (nullable = true)
 # |-- genre_label: string (nullable = true)
 
magd_genre.show(10,40)

# +------------------+----------+
# |          track_id|     genre|
# +------------------+----------+
# |TRESCAC12903CD0F3D|       RnB|
# |TRHWUAP128F932C5E7|       RnB|
# |TRCHGOD128F934514B|   Country|
# |TRCUXLN128F9326A95|  Pop_Rock|
# |TRFGMGS128F92F3A1C|Electronic|
# |TRHZUGZ128F424C9B6|  Pop_Rock|
# |TRENTGL128E0780C8E|  Pop_Rock|
# |TRJHFIY128F42604D8|  Pop_Rock|
# |TRANWBG128F934AC15|  Pop_Rock|
# |TRDLGTK128F9338110|    Reggae|
# +------------------+----------+
# only showing top 10 rows


# Count the distribution of genres
genre_distribution = (
    magd_genre
    .groupBy("genre")
    .agg(
        F.count(F.col('track_id')).alias("count")
    )
)

genre_distribution.repartition(1).write.mode("overwrite").csv("hdfs:///user/sch405/assignment2/msd/features_v1_genre_distribution")
# hdfs dfs -copyToLocal  hdfs:///user/sch405/assignment2/msd/features_v1_genre_distribution/part-00000-f5d54ae2-de01-477c-9281-9f338625b355-c000.csv ~/msd_features_v1_genre_distribution.csv



genre_distribution.count() # 21
genre_distribution.show()


# +--------------+------+
# |         genre| count|
# +--------------+------+
# |          Jazz| 17836|
# |         Blues|  6836|
# |      Pop_Rock|238786|
# |     Classical|   556|
# |        Reggae|  6946|
# |     Religious|  8814|
# |         Vocal|  6195|
# |Easy_Listening|  1545|
# |           Rap| 20939|
# |           RnB| 14335|
# |         Latin| 17590|
# |          Folk|  5865|
# |       Country| 11772|
# |         Stage|  1614|
# |    Electronic| 41075|
# | International| 14242|
# |      Children|   477|
# |   Avant_Garde|  1014|
# |       New Age|  4010|
# | Comedy_Spoken|  2067|
# |       Holiday|   200|
# +--------------+------+



"""genre_distribution = (genre_distribution.sort(F.col(count)))"""


# Audio Similarity Q1 (c)

# -----------------------------------------------------------------------------
# Merge genres dataset and audio features dataset
# -----------------------------------------------------------------------------

features_merge_genres = (
    feature_dataset_v1_drop.
    join(
        magd_genre,
        on="track_id",
        how="left_outer"
    )
)

features_merge_genres.show(10,40)

# +------------------+--------+--------+-------+-------+--------+--------+-------+------+--------+
# |          track_id|  f_0002|  f_0004| f_0005| f_0006|  f_0010|  f_0012| f_0013|f_0014|   genre|
# +------------------+--------+--------+-------+-------+--------+--------+-------+------+--------+
# |TRAAABD128F429CF47|0.001519|0.001557|0.05665| 0.0558| 0.00109|0.002902| 0.1126|0.5304|Pop_Rock|
# |TRAAADZ128F9348C2E|0.001638|0.002156|0.07978|0.07485|0.001605|0.005229| 0.2006|0.4524|    null|
# |TRAAAND12903CD1F1B|0.003171| 0.00271|0.08154|0.07511|0.002439|0.004914| 0.1759|0.5884|    null|
# |TRAAAVL128F93028BC|9.198E-4|0.001212|0.04155|0.04491|5.823E-4| 0.00197|0.07356|0.5935|    null|
# |TRAABPK128F424CFDB|0.004769|0.002363|0.07297| 0.0443|0.003815|0.005105| 0.1974|0.6286|Pop_Rock|
# |TRAABYN12903CFD305|4.123E-4|0.001247|0.04712|0.08117|2.826E-4|0.002039|0.07816|0.5358|    null|
# |TRAACER128F4290F96|0.004536|0.002502|0.09102|0.05125|0.004631|0.006002| 0.2269|0.5374|Pop_Rock|
# |TRAACWF12903CA0AD7|0.003022|0.002332|0.07864|0.05194|0.001936|0.004375| 0.1636|0.5263|    null|
# |TRAADAD128F9336553|0.001876|0.001833|0.06366|0.04211|0.001212|0.003198| 0.1184|0.5068|    null|
# |TRAADRX12903D0EFE8|0.003009|0.001986| 0.0709|0.05733|0.003535|0.006095| 0.2382|0.5164|    null|
# +------------------+--------+--------+-------+-------+--------+--------+-------+------+--------+
# only showing top 10 rows

# Attention: there are a lot of null value of "genre"

# Check the missing value
features_merge_genres.filter(F.col("genre").isNull()).count() # 573999
features_merge_genres.count() # 994615

genre_group = features_merge_genres.groupBy(["genre"]).agg(F.countDistinct("track_id"))

genre_group.show(50,100)

# +--------------+------------------------+
# |         genre|count(DISTINCT track_id)|
# +--------------+------------------------+
# |         Vocal|                    6182|
# |     Religious|                    8780|
# |Easy_Listening|                    1535|
# |    Electronic|                   40665|
# |          Jazz|                   17774|
# |         Blues|                    6801|
# | International|                   14194|
# |      Children|                     463|
# |           RnB|                   14314|
# |           Rap|                   20899|
# |   Avant_Garde|                    1012|
# |         Latin|                   17504|
# |          Folk|                    5789|
# |      Pop_Rock|                  237649|
# |       New Age|                    4000|
# |     Classical|                     555|
# |          null|                  573999|
# |       Country|                   11689|
# |         Stage|                    1613|
# | Comedy_Spoken|                    2067|
# |        Reggae|                    6931|
# |       Holiday|                     200|
# +--------------+------------------------+

# Remove the observations contain 'null' genre

features_merge_genres = (
    features_merge_genres. filter(~F.col("genre").isNull()))

# +------------------+--------+--------+-------+-------+--------+--------+-------+------+----------+
# |          track_id|  f_0002|  f_0004| f_0005| f_0006|  f_0010|  f_0012| f_0013|f_0014|     genre|
# +------------------+--------+--------+-------+-------+--------+--------+-------+------+----------+
# |TRAAABD128F429CF47|0.001519|0.001557|0.05665| 0.0558| 0.00109|0.002902| 0.1126|0.5304|  Pop_Rock|
# |TRAABPK128F424CFDB|0.004769|0.002363|0.07297| 0.0443|0.003815|0.005105| 0.1974|0.6286|  Pop_Rock|
# |TRAACER128F4290F96|0.004536|0.002502|0.09102|0.05125|0.004631|0.006002| 0.2269|0.5374|  Pop_Rock|
# |TRAADYB128F92D7E73|0.002983|0.002261|0.07965|0.06292|0.001982|0.004733| 0.1773|0.5126|      Jazz|
# |TRAAGHM128EF35CF8E|0.005852|0.002468|0.07639|0.06356|0.003312|0.005462| 0.1978| 0.609|Electronic|
# |TRAAGRV128F93526C0|0.002131|0.001519|0.04777|0.04523|0.002354|0.004584| 0.1764|0.5403|  Pop_Rock|
# |TRAAGTO128F1497E3C|0.001236|0.001789|0.06716|0.08579|3.271E-4| 0.00162| 0.0619|0.6142|  Pop_Rock|
# |TRAAHAU128F9313A3D|8.682E-4|9.592E-4| 0.0332|0.06616|8.643E-4|0.002507|0.09907|0.5874|  Pop_Rock|
# |TRAAHEG128E07861C3|  0.0088|0.003949|  0.131|0.05231|0.003686|0.004698| 0.1726|0.6321|       Rap|
# |TRAAHZP12903CA25F4|0.002841|0.002088|0.07393|0.06671|9.622E-4|0.002193|0.08333|0.6052|       Rap|
# +------------------+--------+--------+-------+-------+--------+--------+-------+------+----------+
# only showing top 10 rows

features_merge_genres.count() # 420616



"""
(
features_merge_genres.filter(F.col("genre").isNull() == False)
                     .show(5)
)

(
magd_genre. filter(F.col("track_id")== "TRAAADZ128F9348C2E" ).show(5)
)

# +--------+-----+
# |track_id|genre|
# +--------+-----+
# +--------+-----+


features_merge_genres.count()  # 994180

(features_merge_genres.filter(F.col("genre").isNull() == False).count()) # 0

(feature_dataset_v1.filter(F.col("track_id")==`"TRIJSDD128F93123BE"`).count()) #0

df.withColumn("track_id", F.regexp_replace("track_id", "'",""))
"""


# Audio Similarity Q2 (b)

# -----------------------------------------------------------------------------
# Convert genre column into binary label
# -----------------------------------------------------------------------------

# Add a label column "is _rap"
features_merge_genres_label = (
    features_merge_genres.withColumn("is_rap", 
                                     when(F.col("genre").contains("Rap"), 1)
                                     .otherwise(0)
                                     )
)


features_merge_genres_label.show(10, 100)
# +------------------+--------+--------+-------+-------+--------+--------+-------+------+----------+------+
# |          track_id|  f_0002|  f_0004| f_0005| f_0006|  f_0010|  f_0012| f_0013|f_0014|     genre|is_rap|
# +------------------+--------+--------+-------+-------+--------+--------+-------+------+----------+------+
# |TRAAABD128F429CF47|0.001519|0.001557|0.05665| 0.0558| 0.00109|0.002902| 0.1126|0.5304|  Pop_Rock|     0|
# |TRAABPK128F424CFDB|0.004769|0.002363|0.07297| 0.0443|0.003815|0.005105| 0.1974|0.6286|  Pop_Rock|     0|
# |TRAACER128F4290F96|0.004536|0.002502|0.09102|0.05125|0.004631|0.006002| 0.2269|0.5374|  Pop_Rock|     0|
# |TRAADYB128F92D7E73|0.002983|0.002261|0.07965|0.06292|0.001982|0.004733| 0.1773|0.5126|      Jazz|     0|
# |TRAAGHM128EF35CF8E|0.005852|0.002468|0.07639|0.06356|0.003312|0.005462| 0.1978| 0.609|Electronic|     0|
# |TRAAGRV128F93526C0|0.002131|0.001519|0.04777|0.04523|0.002354|0.004584| 0.1764|0.5403|  Pop_Rock|     0|
# |TRAAGTO128F1497E3C|0.001236|0.001789|0.06716|0.08579|3.271E-4| 0.00162| 0.0619|0.6142|  Pop_Rock|     0|
# |TRAAHAU128F9313A3D|8.682E-4|9.592E-4| 0.0332|0.06616|8.643E-4|0.002507|0.09907|0.5874|  Pop_Rock|     0|
# |TRAAHEG128E07861C3|  0.0088|0.003949|  0.131|0.05231|0.003686|0.004698| 0.1726|0.6321|       Rap|     1|
# |TRAAHZP12903CA25F4|0.002841|0.002088|0.07393|0.06671|9.622E-4|0.002193|0.08333|0.6052|       Rap|     1|
# +------------------+--------+--------+-------+-------+--------+--------+-------+------+----------+------+
# only showing top 10 rows

features_merge_genres_label.show(10, 100)
(
    features_merge_genres_label
    .groupBy("is_rap")
    .count()
    .show(2)
)

# +------+------+
# |is_rap| count|
# +------+------+
# |     0|399717|
# |     1| 20899|
# +------+------+


print(20899 / 399717)
# 0.05228




# Audio Similarity Q2 (c)

# -----------------------------------------------------------------------------
# Splitting
# -----------------------------------------------------------------------------

"""
Given the class imbalance ratio, we recommend measuring the accuracy using the
Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not
meaningful for unbalanced classification.
"""

# Imports

from pyspark.sql.window import *
from pyspark.ml.feature import VectorAssembler

######### stratification without sampling ########

# Helpers

def print_class_balance(data, name):
    N = data.count()
    counts = data.groupBy("is_rap").count().toPandas()
    counts["ratio"] = counts["count"] / N
    print(name)
    print(N)
    print(counts)
    print("")


def print_binary_metrics(predictions, labelCol="is_rap", predictionCol="prediction", rawPredictionCol="rawPrediction"):

    total = predictions.count()
    positive = predictions.filter((col(labelCol) == 1)).count()
    negative = predictions.filter((col(labelCol) == 0)).count()
    nP = predictions.filter((col(predictionCol) == 1)).count()
    nN = predictions.filter((col(predictionCol) == 0)).count()
    TP = predictions.filter((col(predictionCol) == 1) & (col(labelCol) == 1)).count()
    FP = predictions.filter((col(predictionCol) == 1) & (col(labelCol) == 0)).count()
    FN = predictions.filter((col(predictionCol) == 0) & (col(labelCol) == 1)).count()
    TN = predictions.filter((col(predictionCol) == 0) & (col(labelCol) == 0)).count()

    binary_evaluator = BinaryClassificationEvaluator(rawPredictionCol=rawPredictionCol, labelCol=labelCol, metricName="areaUnderROC")
    auroc = binary_evaluator.evaluate(predictions)

    print('actual total:    {}'.format(total))
    print('actual positive: {}'.format(positive))
    print('actual negative: {}'.format(negative))
    print('nP:              {}'.format(nP))
    print('nN:              {}'.format(nN))
    print('TP:              {}'.format(TP))
    print('FP:              {}'.format(FP))
    print('FN:              {}'.format(FN))
    print('TN:              {}'.format(TN))
    print('precision:       {}'.format(TP / (TP + FP)))
    print('recall:          {}'.format(TP / (TP + FN)))
    print('accuracy:        {}'.format((TP + TN) / total))
    print('auroc:           {}'.format(auroc))




######### stratification without downsampling ########


# Imports

import numpy as np

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Helpers

def with_genre_prediction(predictions, threshold, probabilityCol="probability", genrePredictionCol="genrePrediction"):

    def apply_genre_threshold(probability, threshold):
        return int(probability[1] > threshold)

    apply_genre_threshold_udf = udf(lambda x: apply_genre_threshold(x, threshold), IntegerType())

    return predictions.withColumn(genrePredictionCol, apply_genre_threshold_udf(col(probabilityCol)))


# Assemble vector only from the remaining columns


assembler = VectorAssembler(
    inputCols=[col for col in features_merge_genres_label.columns if col.startswith("f")],
    outputCol="Features"
).setHandleInvalid("skip")



features = assembler.transform(features_merge_genres_label).select(["track_id","Features", "is_rap"])

features.cache()
features.show(10, 100)


# +------------------+--------------------------------------------------------------------+------+
# |          track_id|                                                            Features|is_rap|
# +------------------+--------------------------------------------------------------------+------+
# |TRAAABD128F429CF47|   [0.001519,0.001557,0.05665,0.0558,0.00109,0.002902,0.1126,0.5304]|     0|
# |TRAABPK128F424CFDB|  [0.004769,0.002363,0.07297,0.0443,0.003815,0.005105,0.1974,0.6286]|     0|
# |TRAACER128F4290F96| [0.004536,0.002502,0.09102,0.05125,0.004631,0.006002,0.2269,0.5374]|     0|
# |TRAADYB128F92D7E73| [0.002983,0.002261,0.07965,0.06292,0.001982,0.004733,0.1773,0.5126]|     0|
# |TRAAGHM128EF35CF8E|  [0.005852,0.002468,0.07639,0.06356,0.003312,0.005462,0.1978,0.609]|     0|
# |TRAAGRV128F93526C0| [0.002131,0.001519,0.04777,0.04523,0.002354,0.004584,0.1764,0.5403]|     0|
# |TRAAGTO128F1497E3C|  [0.001236,0.001789,0.06716,0.08579,3.271E-4,0.00162,0.0619,0.6142]|     0|
# |TRAAHAU128F9313A3D| [8.682E-4,9.592E-4,0.0332,0.06616,8.643E-4,0.002507,0.09907,0.5874]|     0|
# |TRAAHEG128E07861C3|     [0.0088,0.003949,0.131,0.05231,0.003686,0.004698,0.1726,0.6321]|     1|
# |TRAAHZP12903CA25F4|[0.002841,0.002088,0.07393,0.06671,9.622E-4,0.002193,0.08333,0.6052]|     1|
# +------------------+--------------------------------------------------------------------+------+
# only showing top 10 rows

# Exact stratification using Window (multi-class variant in comments)
temp = (
    features
    .withColumn("id", monotonically_increasing_id())
    .withColumn("Random", rand())
    .withColumn(
        "Row",
        row_number()
        .over(
            Window
            .partitionBy("is_rap")
            .orderBy("Random")
        )
    )
)

training = temp.where(
    ((col("is_rap") == 0) & (col("Row") < 399717 * 0.8)) |
    ((col("is_rap") == 1) & (col("Row") < 20899 * 0.8))
)
training.cache()



test = temp.join(training, on="id", how="left_anti")
test.cache()

training = training.drop("id", "Random", "Row")
test = test.drop("id", "Random", "Row")

print_class_balance(features_merge_genres_label, "is_rap")

# is_rap
# 420616
  # is_rap   count     ratio
# 0      0  399717  0.950313
# 1      1   20899  0.049687

print_class_balance(training, "training")

# training
# 336492
  # is_rap   count     ratio
# 0      0  319773  0.950314
# 1      1   16719  0.049686

print_class_balance(test, "test")

# test
# 84124
  # is_rap  count     ratio
# 0      0  79944  0.950311
# 1      1   4180  0.049689


# Check stuff is cached
features.cache()
training.cache()
test.cache()
training.show(10,100)

# +------------------+--------------------------------------------------------------------+------+
# |          track_id|                                                            Features|is_rap|
# +------------------+--------------------------------------------------------------------+------+
# |TRCMDPN128F429224B|  [0.002505,0.002473,0.08191,0.04813,0.001688,0.003712,0.135,0.5981]|     1|
# |TRGCNWO128F4281326|      [0.004163,0.00293,0.0942,0.1001,0.002596,0.004786,0.1698,0.39]|     1|
# |TRIEISM128F934628E|   [0.008898,0.003472,0.1156,0.07683,0.00401,0.004629,0.1742,0.6599]|     1|
# |TRUVSPE12903CC7F57|  [0.003888,0.001917,0.06024,0.08026,0.002288,0.004667,0.1765,0.663]|     1|
# |TRWXXFU12903D04910| [0.006266,0.003251,0.1062,0.08191,0.002027,0.002653,0.09732,0.5236]|     1|
# |TRDCJAG12903CD0D46|    [0.008154,0.00341,0.1074,0.04585,0.004771,0.005546,0.1966,0.597]|     1|
# |TRRSFEA128F92E2391|   [0.01581,0.005832,0.1817,0.06867,0.007075,0.005748,0.2102,0.6773]|     1|
# |TRDMZGQ12903CC5AF8|   [0.009711,0.00348,0.1118,0.03548,0.008346,0.008154,0.3029,0.5368]|     1|
# |TRSSJND128F9307B99|  [0.005644,0.003607,0.1114,0.04687,0.004806,0.007183,0.2595,0.5246]|     1|
# |TRMGQBP128F931C013|[4.015E-4,0.001278,0.04629,0.09911,2.833E-4,0.001842,0.06989,0.5414]|     1|
# +------------------+--------------------------------------------------------------------+------+
# only showing top 10 rows



# -----------
# No sampling
# -----------


lr = LogisticRegression(featuresCol='Features', labelCol='is_rap')
lr_model = lr.fit(training)
predictions = lr_model.transform(test)
predictions.cache()

print_binary_metrics(predictions)


# actual total:    84124
# actual positive: 4180
# actual negative: 79944
# nP:              683
# nN:              83441
# TP:              202
# FP:              481
# FN:              3978
# TN:              79463
# precision:       0.2957540263543192
# recall:          0.04832535885167464
# accuracy:        0.9469949122723599
# auroc:           0.8518574560206574



# ------------
# Downsampling
# ------------

training_downsampled = (
    training
    .withColumn("Random", rand())
    .where((col("is_rap") != 0) | ((col("is_rap") == 0) & (col("Random") < 2 * (20899 / 399717))))
)
training_downsampled.cache()

print_class_balance(training_downsampled, "training_downsampled")

# training_downsampled
# 50125
   # is_rap  count     ratio
# 0       0  33406  0.666454
# 1       1  16719  0.333546


# -----------------------------------------
# Downsampling & LogisticRegression
# -----------------------------------------
lr = LogisticRegression(featuresCol='Features', labelCol='is_rap')
lr_model = lr.fit(training_downsampled)
predictions = lr_model.transform(test)
predictions.cache()

print_binary_metrics(predictions)

# actual total:    84124
# actual positive: 4180
# actual negative: 79944
# nP:              10460
# nN:              73664
# TP:              2502
# FP:              7958
# FN:              1678
# TN:              71986
# precision:       0.23919694072657743
# recall:          0.5985645933014354
# accuracy:        0.8854548048119443
# auroc:           0.8591593646054698




# -----------------------------------------
# Downsampling & RandomForest
# -----------------------------------------
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(labelCol="is_rap",
                            featuresCol="Features",
                            numTrees = 20,
                            maxDepth = 4,
                            maxBins = 2)
                            
# Train model with Training_downsampled Data
rfModel = rf.fit(training_downsampled)
predictions = rfModel.transform(test)
predictions.filter(predictions['prediction'] == 0) \
    .select("Features","is_rap","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)

print_class_balance(training_downsampled, "training_downsampled")

# training_downsampled
# 50125
   # is_rap  count     ratio
# 0       0  33406  0.666454
# 1       1  16719  0.333546

print_binary_metrics(predictions)


# actual total:    84124
# actual positive: 4180
# actual negative: 79944
# nP:              24266
# nN:              59858
# TP:              3010
# FP:              21256
# FN:              1170
# TN:              58688
# precision:       0.12404186928212313
# recall:          0.7200956937799043
# accuracy:        0.7334173363130617
# auroc:           0.7771358835754406


# -----------------------------------------
# Downsampling &  Decision Tree
# -----------------------------------------

from pyspark.ml.classification import DecisionTreeClassifier

dt = DecisionTreeClassifier(labelCol="is_rap", featuresCol="Features")

# Train model with Training_downsampled Data
dtModel = dt.fit(training_downsampled)

predictions = rfModel.transform(test)

print_class_balance(training_downsampled, "training_downsampled")


# training_downsampled
# 50125
   # is_rap  count     ratio
# 0       0  33406  0.666454
# 1       1  16719  0.333546


print_binary_metrics(predictions)

# actual total:    84124
# actual positive: 4180
# actual negative: 79944
# nP:              24266
# nN:              59858
# TP:              3010
# FP:              21256
# FN:              1170
# TN:              58688
# precision:       0.12404186928212313
# recall:          0.7200956937799043
# accuracy:        0.7334173363130617
# auroc:           0.7607045221726978


# -----------------------------------------
# Tune hyperparameters
# -----------------------------------------

# Create ParamGrid for Cross Validation

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="is_rap")
evaluator.evaluate(predictions)


# -----------------------------------------
# Tune hyperparameters & LogisticRegression
# -----------------------------------------

paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.1, 0.5]) # regularization parameter
             .addGrid(lr.elasticNetParam, [0.0, 0.2]) # Elastic Net Parameter (Ridge = 0)
#            .addGrid(model.maxIter, [10, 20, 50]) #Number of iterations
#            .addGrid(idf.numFeatures, [10, 100, 1000]) # Number of features
             .build())
# Create 5-fold CrossValidator
cv = CrossValidator(estimator=lr, 
                    estimatorParamMaps=paramGrid, 
                    evaluator=evaluator,
                    numFolds=5)
cvModel = cv.fit(training_downsampled)

predictions = cvModel.transform(test)
# Evaluate best model
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="is_rap")

evaluator.evaluate(predictions) # 0.917789368766363

print_class_balance(training_downsampled, "training_downsampled")

# training_downsampled
# 50125
   # is_rap  count     ratio
# 0       0  33406  0.666454
# 1       1  16719  0.333546

print_binary_metrics(predictions) 

# actual total:    84124
# actual positive: 4180
# actual negative: 79944
# nP:              7946
# nN:              76178
# TP:              2027
# FP:              5919
# FN:              2153
# TN:              74025
# precision:       0.2550969041026932
# recall:          0.48492822966507176
# accuracy:        0.9040464076838952
# auroc:           0.8218023564461677



# -----------------------------------------
# Multiclass classification
# -----------------------------------------
from pyspark.ml.evaluation import MulticlassClassificationEvaluator as MCE

# Add a label column "Class"
features_merge_genres_label = (
    features_merge_genres.withColumn("Class", 
                                     when(F.col("genre").contains("Rap"), 1)
                                     .when(F.col("genre").contains("Jazz"),2)
                                     .when(F.col("genre").contains("Blues"),3)
                                     .when(F.col("genre").contains("Pop_Rock"),4)
                                     .when(F.col("genre").contains("Classical"),5)
                                     .when(F.col("genre").contains("Reggae"),6)
                                     .when(F.col("genre").contains("Religious"),7)
                                     .when(F.col("genre").contains("Vocal"),8)
                                     .when(F.col("genre").contains("Easy_Listening"),9)
                                     .when(F.col("genre").contains("RnB"),10)
                                     .when(F.col("genre").contains("Latin"),11)
                                     .when(F.col("genre").contains("Folk"),12)
                                     .when(F.col("genre").contains("Country"),13)
                                     .when(F.col("genre").contains("Stage"),14)
                                     .when(F.col("genre").contains("Electronic"),15)
                                     .when(F.col("genre").contains("International"),16)
                                     .when(F.col("genre").contains("Children"),17)
                                     .when(F.col("genre").contains("Avant_Garde"),18)
                                     .when(F.col("genre").contains("New Age"),19)
                                     .when(F.col("genre").contains("Comedy_Spoken"),20)
                                     .when(F.col("genre").contains("Holiday"),21)
                                     )
)

features_merge_genres_label.show(10, 100)

# +------------------+--------+--------+-------+-------+--------+--------+------+------+----------+-----+
# |          track_id|  f_0002|  f_0004| f_0005| f_0006|  f_0010|  f_0012|f_0013|f_0014|     genre|Class|
# +------------------+--------+--------+-------+-------+--------+--------+------+------+----------+-----+
# |TRAAABD128F429CF47|0.001519|0.001557|0.05665| 0.0558| 0.00109|0.002902|0.1126|0.5304|  Pop_Rock|    4|
# |TRAAAEF128F4273421|9.944E-4|0.001172|0.04344|0.05534|8.953E-4|0.002842|0.1133|0.5422|  Pop_Rock|    4|
# |TRAAAIR128F1480971|0.009601|0.003531| 0.1192|0.05424|0.006124|0.007223|0.2708|0.5909|       RnB|   10|
# |TRAAAMO128F1481E7F|0.001631|0.001468|0.04735|0.05611|0.001698|0.004041|0.1478|0.5223| Religious|    7|
# |TRAACLG128F4276511| 0.01205|0.005275| 0.1502|0.02528|0.008857|0.008184|0.2824| 0.607|Electronic|   15|
# |TRAADMZ128F422F2F8|0.004146|0.002341|0.07626|0.03364|0.003002|0.004696|0.1716|0.5663|  Pop_Rock|    4|
# |TRAADYB128F92D7E73|0.002983|0.002261|0.07965|0.06292|0.001982|0.004733|0.1773|0.5126|      Jazz|    2|
# |TRAAFTE128F429545F|0.003126|0.002083|0.07685|0.06178|0.001778|  0.0037|0.1433|0.5693|  Pop_Rock|    4|
# |TRAAGAV128F4241242|0.004237|0.001988|0.06203|0.04199|0.004097|0.005174|0.2041|0.5961|  Pop_Rock|    4|
# |TRAAGCG128F421CC9F|0.002427|0.002146|0.07174|0.06791|0.001557|0.003849|0.1413|0.4322|     Latin|   11|
# +------------------+--------+--------+-------+-------+--------+--------+------+------+----------+-----+
# only showing top 10 rows



features_merge_genres_label.show(10, 100)
(
    features_merge_genres_label
    .groupBy("Class")
    .count()
    .show(22)
)


# +-----+------+
# |Class| count|
# +-----+------+
# |   14|  1613|
# |   18|  1012|
# |   19|  4000|
# |    6|  6931|
# |   20|  2067|
# |    8|  6182|
# |    7|  8780|
# |    3|  6801|
# |   21|   200|
# |   13| 11689|
# |   15| 40665|
# |   16| 14194|
# |    4|237649|
# |   10| 14314|
# |    2| 17774|
# |    5|   555|
# |    1| 20899|
# |   12|  5789|
# |   17|   463|
# |    9|  1535|
# |   11| 17504|
# +-----+------+


assembler = VectorAssembler(
    inputCols=[col for col in features_merge_genres_label.columns if col.startswith("f")],
    outputCol="Features"
).setHandleInvalid("skip")



features = assembler.transform(features_merge_genres_label).select(["track_id","Features", "Class"])

features.cache()
features.show(10, 100)

# +------------------+-------------------------------------------------------------------+-----+
# |          track_id|                                                           Features|Class|
# +------------------+-------------------------------------------------------------------+-----+
# |TRAAABD128F429CF47|  [0.001519,0.001557,0.05665,0.0558,0.00109,0.002902,0.1126,0.5304]|    4|
# |TRAAAEF128F4273421|[9.944E-4,0.001172,0.04344,0.05534,8.953E-4,0.002842,0.1133,0.5422]|    4|
# |TRAAAIR128F1480971| [0.009601,0.003531,0.1192,0.05424,0.006124,0.007223,0.2708,0.5909]|   10|
# |TRAAAMO128F1481E7F|[0.001631,0.001468,0.04735,0.05611,0.001698,0.004041,0.1478,0.5223]|    7|
# |TRAACLG128F4276511|   [0.01205,0.005275,0.1502,0.02528,0.008857,0.008184,0.2824,0.607]|   15|
# |TRAADMZ128F422F2F8|[0.004146,0.002341,0.07626,0.03364,0.003002,0.004696,0.1716,0.5663]|    4|
# |TRAADYB128F92D7E73|[0.002983,0.002261,0.07965,0.06292,0.001982,0.004733,0.1773,0.5126]|    2|
# |TRAAFTE128F429545F|  [0.003126,0.002083,0.07685,0.06178,0.001778,0.0037,0.1433,0.5693]|    4|
# |TRAAGAV128F4241242|[0.004237,0.001988,0.06203,0.04199,0.004097,0.005174,0.2041,0.5961]|    4|
# |TRAAGCG128F421CC9F|[0.002427,0.002146,0.07174,0.06791,0.001557,0.003849,0.1413,0.4322]|   11|
# +------------------+-------------------------------------------------------------------+-----+
# only showing top 10 rows


# Exact stratification using Window (multi-class variant in comments)


# Helpers

def print_class_balance(data, name):
    N = data.count()
    counts = data.groupBy("Class").count().toPandas()
    counts["ratio"] = counts["count"] / N
    print(name)
    print(N)
    print(counts)
    print("")
    
    


temp = (
    features
    .withColumn("id", monotonically_increasing_id())
    .withColumn("Random", rand())
    .withColumn(
        "Row",
        row_number()
        .over(
            Window
            .partitionBy("Class")
            .orderBy("Random")
        )
    )
)
"""
training = temp.where(
    ((col("Class") == 1) & (col("Row") < 20899 * 0.8)) |
    ((col("Class") == 2) & (col("Row") < 17774 * 0.8)) |
    ((col("Class") == 3) & (col("Row") < 6801 * 0.8)) |
    ((col("Class") == 4) & (col("Row") < 237649 * 0.8)) | 
    ((col("Class") == 5) & (col("Row") < 399717 * 0.8)) |
    ((col("Class") == 6) & (col("Row") < 6931 * 0.8)) |
    ((col("Class") == 7) & (col("Row") < 8780 * 0.8)) |
    ((col("Class") == 8) & (col("Row") < 6182 * 0.8)) |
    ((col("Class") == 9) & (col("Row") < 1535 * 0.8)) | 
    ((col("Class") == 10) & (col("Row") < 14314 * 0.8)) |
    ((col("Class") == 11) & (col("Row") < 17504 * 0.8)) | 
    ((col("Class") == 12) & (col("Row") < 5789 * 0.8)) |
    ((col("Class") == 13) & (col("Row") < 11689 * 0.8)) |
    ((col("Class") == 14) & (col("Row") < 1613 * 0.8)) |
    ((col("Class") == 15) & (col("Row") < 40665 * 0.8)) |
    ((col("Class") == 16) & (col("Row") < 14194 * 0.8)) | 
    ((col("Class") == 17) & (col("Row") < 463 * 0.8)) |
    ((col("Class") == 18) & (col("Row") < 1012 * 0.8)) |
    ((col("Class") == 19) & (col("Row") < 4000 * 0.8)) |
    ((col("Class") == 20) & (col("Row") < 2067 * 0.8)) |
    ((col("Class") == 21) & (col("Row") < 200 * 0.8)) 
)
training.cache()
training.show(10,100)
# +------------------+--------------------------------------------------------------------+-----+-----------+---------------------+---+
# |          track_id|                                                            Features|Class|         id|               Random|Row|
# +------------------+--------------------------------------------------------------------+-----+-----------+---------------------+---+
# |TRBFGKU128F14ACF03| [4.109E-4,8.021E-4,0.02586,0.05477,1.907E-4,0.001016,0.03645,0.578]|   14|60129543768| 0.002019489582244516|  1|
# |TRUCHSB128F93526B5|  [0.008982,0.003162,0.1039,0.07135,0.003995,0.004255,0.1568,0.6671]|   14|94489307683| 0.002756944229875047|  2|
# |TRLNGGN128F42756BF|  [0.002674,0.002171,0.07238,0.07497,0.00118,0.003457,0.1265,0.5848]|   14|51539623341| 0.003189916694055883|  3|
# |TRYTIAY128F42926F3|[5.138E-4,0.001175,0.04247,0.05181,4.623E-4,0.002354,0.08987,0.5608]|   14|60129575604|0.0035103452633515886|  4|
# |TRIETGR128F932E959| [0.001149,0.001712,0.06316,0.06828,8.954E-4,0.003018,0.1152,0.5554]|   14|68719487623|0.0035803650018962907|  5|
# |TRTKKGY12903CD57BF|     [0.0012,0.00178,0.06759,0.0757,0.001097,0.003681,0.1429,0.5314]|   14|51539633895| 0.003844871418016149|  6|
# |TRUMZMV128F426F39C|    [4.627E-5,3.597E-4,0.01288,0.07497,3.0E-5,5.668E-4,0.0211,0.561]|   14|25769831502| 0.003881911731826171|  7|
# |TRZQMGV128F4221006| [7.967E-4,0.001429,0.05161,0.05449,9.169E-4,0.003832,0.1457,0.5059]|   14|25769838400| 0.004387503289432382|  8|
# |TRKBEHN128F933C43A|  [0.00133,0.001613,0.05757,0.06013,9.459E-4,0.002897,0.1107,0.5273]|   14|42949686592| 0.005227422061659048|  9|
# |TRXGSHC128F4272A41|  [0.001983,0.001931,0.06453,0.04932,0.001145,0.002837,0.1053,0.544]|   14|17179900421| 0.005832145102972586| 10|
# +------------------+--------------------------------------------------------------------+-----+-----------+---------------------+---+
# only showing top 10 rows

"""

training = temp
for c in classes:
    training = training.where((col("Class") != i) | (col("Row") < class_counts[i] * 0.8))

training.cache()
training.show(10, 100)

# +------------------+--------------------------------------------------------------------+-----+-----------+---------------------+---+
# |          track_id|                                                            Features|Class|         id|               Random|Row|
# +------------------+--------------------------------------------------------------------+-----+-----------+---------------------+---+
# |TRBFGKU128F14ACF03| [4.109E-4,8.021E-4,0.02586,0.05477,1.907E-4,0.001016,0.03645,0.578]|   14|60129543768| 0.002019489582244516|  1|
# |TRUCHSB128F93526B5|  [0.008982,0.003162,0.1039,0.07135,0.003995,0.004255,0.1568,0.6671]|   14|94489307683| 0.002756944229875047|  2|
# |TRLNGGN128F42756BF|  [0.002674,0.002171,0.07238,0.07497,0.00118,0.003457,0.1265,0.5848]|   14|51539623341| 0.003189916694055883|  3|
# |TRYTIAY128F42926F3|[5.138E-4,0.001175,0.04247,0.05181,4.623E-4,0.002354,0.08987,0.5608]|   14|60129575604|0.0035103452633515886|  4|
# |TRIETGR128F932E959| [0.001149,0.001712,0.06316,0.06828,8.954E-4,0.003018,0.1152,0.5554]|   14|68719487623|0.0035803650018962907|  5|
# |TRTKKGY12903CD57BF|     [0.0012,0.00178,0.06759,0.0757,0.001097,0.003681,0.1429,0.5314]|   14|51539633895| 0.003844871418016149|  6|
# |TRUMZMV128F426F39C|    [4.627E-5,3.597E-4,0.01288,0.07497,3.0E-5,5.668E-4,0.0211,0.561]|   14|25769831502| 0.003881911731826171|  7|
# |TRZQMGV128F4221006| [7.967E-4,0.001429,0.05161,0.05449,9.169E-4,0.003832,0.1457,0.5059]|   14|25769838400| 0.004387503289432382|  8|
# |TRKBEHN128F933C43A|  [0.00133,0.001613,0.05757,0.06013,9.459E-4,0.002897,0.1107,0.5273]|   14|42949686592| 0.005227422061659048|  9|
# |TRXGSHC128F4272A41|  [0.001983,0.001931,0.06453,0.04932,0.001145,0.002837,0.1053,0.544]|   14|17179900421| 0.005832145102972586| 10|
# +------------------+--------------------------------------------------------------------+-----+-----------+---------------------+---+
# only showing top 10 rows

test = temp.join(training, on="track_id", how="left_anti")
test.cache()

training = training.drop("track_id", "Random", "Row")
test = test.drop("track_id", "Random", "Row")

print_class_balance(features_merge_genres_label, "Class")
# Class
# 420616
    # Class   count     ratio
# 0      14    1613  0.003835
# 1      18    1012  0.002406
# 2      20    2067  0.004914
# 3       6    6931  0.016478
# 4      19    4000  0.009510
# 5       7    8780  0.020874
# 6       3    6801  0.016169
# 7       8    6182  0.014697
# 8      21     200  0.000475
# 9      13   11689  0.027790
# 10     15   40665  0.096680
# 11     16   14194  0.033746
# 12      4  237649  0.565002
# 13      2   17774  0.042257
# 14     10   14314  0.034031
# 15      5     555  0.001319
# 16      1   20899  0.049687
# 17     12    5789  0.013763
# 18      9    1535  0.003649
# 19     17     463  0.001101
# 20     11   17504  0.041615


print_class_balance(training, "training")

# training
# 336593
    # Class   count     ratio
# 0      14    1290  0.003833
# 1      18     809  0.002403
# 2       6    5544  0.016471
# 3      19    3199  0.009504
# 4      20    1653  0.004911
# 5       3    5440  0.016162
# 6       7    7023  0.020865
# 7       8    4945  0.014691
# 8      21     159  0.000472
# 9      13    9351  0.027781
# 10     15   32531  0.096648
# 11     16   11355  0.033735
# 12      2   14219  0.042244
# 13      4  190119  0.564833
# 14      5     555  0.001649
# 15     10   11451  0.034020
# 16      1   16719  0.049671
# 17     12    4631  0.013758
# 18      9    1227  0.003645
# 19     17     370  0.001099
# 20     11   14003  0.041602


print_class_balance(test, "test")

# test
# 84023
    # Class  count     ratio
# 0      14    323  0.003844
# 1      18    203  0.002416
# 2       6   1387  0.016507
# 3      19    801  0.009533
# 4      20    414  0.004927
# 5       7   1757  0.020911
# 6       8   1237  0.014722
# 7       3   1361  0.016198
# 8      21     41  0.000488
# 9      13   2338  0.027826
# 10     15   8134  0.096807
# 11     16   2839  0.033788
# 12      4  47530  0.565678
# 13     10   2863  0.034074
# 14      2   3555  0.042310
# 15      1   4180  0.049748
# 16     12   1158  0.013782
# 17     17     93  0.001107
# 18      9    308  0.003666
# 19     11   3501  0.041667


# Check stuff is cached
features.cache()
training.cache()
test.cache()
training.show(10,100)

# +--------------------------------------------------------------------+-----+
# |                                                            Features|Class|
# +--------------------------------------------------------------------+-----+
# | [4.109E-4,8.021E-4,0.02586,0.05477,1.907E-4,0.001016,0.03645,0.578]|   14|
# |  [0.008982,0.003162,0.1039,0.07135,0.003995,0.004255,0.1568,0.6671]|   14|
# |  [0.002674,0.002171,0.07238,0.07497,0.00118,0.003457,0.1265,0.5848]|   14|
# |[5.138E-4,0.001175,0.04247,0.05181,4.623E-4,0.002354,0.08987,0.5608]|   14|
# | [0.001149,0.001712,0.06316,0.06828,8.954E-4,0.003018,0.1152,0.5554]|   14|
# |     [0.0012,0.00178,0.06759,0.0757,0.001097,0.003681,0.1429,0.5314]|   14|
# |    [4.627E-5,3.597E-4,0.01288,0.07497,3.0E-5,5.668E-4,0.0211,0.561]|   14|
# | [7.967E-4,0.001429,0.05161,0.05449,9.169E-4,0.003832,0.1457,0.5059]|   14|
# |  [0.00133,0.001613,0.05757,0.06013,9.459E-4,0.002897,0.1107,0.5273]|   14|
# |  [0.001983,0.001931,0.06453,0.04932,0.001145,0.002837,0.1053,0.544]|   14|
# +--------------------------------------------------------------------+-----+
# only showing top 10 rows


# --------------------------------------
# Muticlass classification Downsampling
# --------------------------------------


train_class_count = (training.groupBy(F.col("Class")).agg(F.countDistinct(F.col("Features"))))


for c in classes:
    training_downsampled = (
    training
    .withColumn("Random", rand())
    .where((col("Class") != i) | ((col("Class") == i) & (col("Random") < 21 * ( int(train_class_count.filter(F.col("Class") == i))/ 336593))))
    )
training_downsampled.cache()

print_class_balance(training_downsampled, "training_downsampled")



class_counts = (
    features
    .groupBy("Class")
    .count()
    .toPandas()
    .set_index("Class")["count"]
    .to_dict()
)
classes = sorted(class_counts.keys())



print_class_balance(training_downsampled, "training_downsampled")




# -----------------------------------------------------------
# Muticlass classification Downsampling & LogisticRegression
# -----------------------------------------------------------
lr = LogisticRegression(featuresCol='Features', labelCol='Class')
lr_model = lr.fit(training)
predictions = lr_model.transform(test)

predictions.cache()

# ----------------------------------------------
# Muticlass classification & performance metrics
# ----------------------------------------------

"""
from pycm import *

y_actu =[]
y_pred = []
cm = ConfusionMatrix(actual_vector=y_actu, predict_vector=y_pred) # Create CM From Data
cm.classes
cm.table
print(cm)
cm.matrix()
cm.normalized_matrix()

"""

evaluator_metrics = ["f1", "WeightedPrecision", "weightedRecall", "accuracy"]

def print_multiClass_metrics(predictions, labelCol="Class", predictionCol="prediction", rawPredictionCol="rawPrediction"):
    total = predictions.count()
    for i in evaluatro_metrics:
        metric_value = (MCE.evaluate(predictions,{MCE.metricName: i})
    )
    print(f"{metri:25s}{metric_value}")
    
    

print_multiClass_metrics(predictions)






# Song recommendations Q1 (a)

# -----------------------------------------------------------------------------
# Properties of user-song play information
# -----------------------------------------------------------------------------


def get_user_counts(triplets):
    return (
        triplets
        .groupBy("user_id")
        .agg(
            F.count(col("song_id")).alias("song_count"),
            F.sum(col("plays")).alias("play_count"),
        )
        .orderBy(col("play_count").desc())
    )

def get_song_counts(triplets):
    return (
        triplets
        .groupBy("song_id")
        .agg(
            F.count(col("user_id")).alias("user_count"),
            F.sum(col("plays")).alias("play_count"),
        )
        .orderBy(col("play_count").desc())
    )

# User statistics

user_counts = (
    triplets_not_mismatched
    .groupBy("user_id")
    .agg(
        F.count(col("song_id")).alias("song_count"),
        F.sum(col("plays")).alias("play_count"),
    )
    .orderBy(col("play_count").desc())
)
user_counts.cache()
user_counts.count() # 1019318


user_counts.show(10, False)
user_counts.repartition(1).write.mode("overwrite").csv("hdfs:///user/sch405/assignment2/msd/userCount")
# hdfs dfs -copyToLocal  hdfs:///user/sch405/assignment2/msd/userCount/part-00000-*.csv ~/msd_userCount.csv

# +----------------------------------------+----------+----------+
# |user_id                                 |song_count|play_count|
# +----------------------------------------+----------+----------+
# |093cb74eb3c517c5179ae24caf0ebec51b24d2a2|195       |13074     |
# |119b7c88d58d0c6eb051365c103da5caf817bea6|1362      |9104      |
# |3fa44653315697f42410a30cb766a4eb102080bb|146       |8025      |
# |a2679496cd0af9779a92a13ff7c6af5c81ea8c7b|518       |6506      |
# |d7d2d888ae04d16e994d6964214a1de81392ee04|1257      |6190      |
# |4ae01afa8f2430ea0704d502bc7b57fb52164882|453       |6153      |
# |b7c24f770be6b802805ac0e2106624a517643c17|1364      |5827      |
# |113255a012b2affeab62607563d03fbdf31b08e7|1096      |5471      |
# |99ac3d883681e21ea68071019dba828ce76fe94d|939       |5385      |
# |6d625c6557df84b60d90426c0116138b617b9449|1307      |5362      |
# +----------------------------------------+----------+----------+

# There are some bad pattern: the first user only listen 195 songs, but played 13074 times, 
# It's almost 67 times per song. This is not a normal actural user (it is not good for training)
# This kind of outliers are suggested to remove.

statistics = (
    user_counts
    .select("song_count", "play_count")
    .describe()
    .toPandas()
    .set_index("summary")
    .rename_axis(None)
    .T
)
print(statistics)

#               count                mean              stddev min    max
# song_count  1019318   44.92720721109605   54.91113199747355   3   4316
# play_count  1019318  128.82423149596102  175.43956510304616   3  13074

user_counts.approxQuantile("song_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05)
user_counts.approxQuantile("play_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05)

# [3.0, 20.0, 32.0,  58.0,  4316.0]
# [3.0, 35.0, 71.0, 173.0, 13074.0]

# This statistic result suggest the distribution of the number of played songs and the distribution of the times of play.
# We can focus on min value, the very less, one user only listened 3 songs; one user only played 3 times.
# Average: each user listen 32 songs and plays 71 times. Average, 2-3 times per song.


# Song statistics

song_counts = (
    triplets_not_mismatched
    .groupBy("song_id")
    .agg(
        F.count(col("user_id")).alias("user_count"),
        F.sum(col("plays")).alias("play_count"),
    )
    .orderBy(col("play_count").desc())
)
song_counts.cache()
song_counts.count()  # 378310

song_counts.show(10, False)



# +------------------+----------+----------+
# |song_id           |user_count|play_count|
# +------------------+----------+----------+
# |SOBONKR12A58A7A7E0|84000     |726885    |
# |SOSXLTC12AF72A7F54|80656     |527893    |
# |SOEGIYH12A6D4FC0E3|69487     |389880    |
# |SOAXGDH12A8C13F8A1|90444     |356533    |
# |SONYKOW12AB01849C9|78353     |292642    |
# |SOPUCYA12A8C13A694|46078     |274627    |
# |SOUFTBI12AB0183F65|37642     |268353    |
# |SOVDSJC12A58A7A271|36976     |244730    |
# |SOOFYTN12A6D4F9B35|40403     |241669    |
# |SOHTKMO12AB01843B0|46077     |236494    |
# +------------------+----------+----------+
# only showing top 10 rows



statistics = (
    song_counts
    .select("user_count", "play_count")
    .describe()
    .toPandas()
    .set_index("summary")
    .rename_axis(None)
    .T
)
print(statistics)

#              count                mean             stddev min     max
# user_count  378310  121.05181200602681  748.6489783736941   1   90444
# play_count  378310   347.1038513388491  2978.605348838212   1  726885

song_counts.approxQuantile("user_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05)
song_counts.approxQuantile("play_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05)

# [1.0, 4.0, 15.0, 44.0, 90444.0]    # output will change everytime, why?
# [1.0, 7.0, 30.0, 111.0, 726885.0]  # output will change everytime, why?


# From the min, there are song only played by one user, only played once. So this song definitely not popular 
# and obviousely it should not be put in the recommend list.
# We are doing recommend songs to user, not find users for paticular song. So we want to focus on the most efficient way to 
# recommen the most possible song that the user will interested in. Under the assumption that one user listen to one song, he must at least listen to two other songs.
# So we donnot want invovle these low frequency count. We want to make our ideal set smaller, make model faster to train.
# The low frequency users may need other particular business strategy.

# If we want to make it smaller, through out users and songs below the some kind of threshold, based on the value they been evaluate with. 
# We can use these quantail output to do that. 
# For example keep 50% users and 50% songs. It will reduce our overall size of the dataset.


triplets_not_mismatched.count()   # 45795111


# If we want to make it smaller, through out users and songs below the some kind of threshold, based on the value they been evaluate with. 
# We can use these quantail output to do that. 
# For example keep 50% users and 50% songs. It will reduce our overall size of the dataset.



# -----------------------------------------------------------------------------
# Visualisation of song popularity distribution
# -----------------------------------------------------------------------------


import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of song popularity
# Define the x, y value

y = [val.play_count for val in song_counts.select(col("play_count")).collect()]
x = [val.song_id for val in song_counts.select(col("song_id")).collect()]

Output_plot = sns.distplot(np.asarray(y), rug=False, hist = False)
plt.tight_layout()
plt.savefig("song_pop_dist.png", bbox_inches = "tight")
plt.close()

# Distribution of user activity

y = [val.play_count for val in user_counts.select(col("play_count")).collect()]
x = [val.user_id for val in user_counts.select(col("user_id")).collect()]

Output_plot = sns.distplot(np.asarray(y), rug=False, hist = False)
plt.tight_layout()
plt.savefig("user_act_dist.png", bbox_inches = "tight")
plt.close()



# -----------------------------------------------------------------------------
# Limiting
# -----------------------------------------------------------------------------

# Define the limilation shreshold for user and song.
user_song_count_threshold = 10 #  Remove users
song_user_count_threshold = 5  #  Remove songs

triplets_limited = triplets_not_mismatched

# for i in range(0, 10):

# Only keep the users who listen to the number of songs are more than the threshold.
triplets_limited = (
        triplets_limited
        .join(
            triplets_limited.groupBy("user_id").count().where(col("count") > user_song_count_threshold).select("user_id"),
            on="user_id",
            how="inner"
        )
    )


# Only keep the songs which been listend by the number of user are more than the threshold.
triplets_limited = (
        triplets_limited
        .join(
            triplets_limited.groupBy("song_id").count().where(col("count") > user_song_count_threshold).select("song_id"),
            on="song_id",
            how="inner"
        )
    )

triplets_limited.cache()
triplets_limited.count()  # 44,296,508 



print(triplets_not_mismatched.count() - triplets_limited.count()) # 1,498,603 (same as the first time)
print(triplets_limited.count() / triplets_not_mismatched.count()) # 0.9672759172916952 (same as the first time)


(
    triplets_limited
    .agg(
        countDistinct(col("user_id")).alias('user_count'),
        countDistinct(col("song_id")).alias('song_count')
    )
    .toPandas()
    .T
    .rename(columns={0: "value"})
)

#              value
# user_count  939471 / 1019318 = 0.9216
# song_count  206894 /  378310 = 0.5469

print(get_user_counts(triplets_limited).approxQuantile("song_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05))
print(get_song_counts(triplets_limited).approxQuantile("user_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05))
# [1.0, 19.0, 36.0, 62.0, 3842.0]          (first output [ 7.0, 23.0, 34.0,  61.0,  3839.0]) 
# [11.0, 26.0, 49.0, 172.0, 86640.0]       (first output [11.0, 26.0, 53.0, 169.0, 86546.0])



# Try Another reasonable large threshold
user_song_count_threshold = 45 #  Remove users
song_user_count_threshold = 5  #  Remove songs

triplets_limited = triplets_not_mismatched



# Only keep the users who listen to the number of songs are more than the threshold.
triplets_limited = (
    triplets_limited
    .join(
        triplets_limited.groupBy("user_id").count().where(col("count") > user_song_count_threshold).select("user_id"),
        on="user_id",
        how="inner"
        )
)


# Only keep the songs which been listend by the number of user are more than the threshold.
triplets_limited = (
    triplets_limited
    .join(
        triplets_limited.groupBy("song_id").count().where(col("count") > user_song_count_threshold).select("song_id"),
        on="song_id",
        how="inner"
        )
)

triplets_limited.cache()
triplets_limited.count()   #  27,654,244


print(triplets_not_mismatched.count() - triplets_limited.count()) # 18,140,867

print(triplets_limited.count() / triplets_not_mismatched.count()) # 0.6038689151774301


# Clever filter algorithm will be more effective when having more information per user. 
# Maximaze the information per user withought throng out too many users. Good result with less noise. better set of inputs.
# Take less time to evaluate this.
# Consider the consquence, even we through away some users, not including them in the algorithm, therefore we cannot make recommendation for these users.
# These 40% users don't have enough reactivation anyway, so these less information cannot ensure the model will work for them, maybe other easy proaches may 
# perform batter on these users rather than the clever filter. (business logic alone, popularity based recommondation, chatagory recommandation

 

(
    triplets_limited
    .agg(
        countDistinct(col("user_id")).alias('user_count'),
        countDistinct(col("song_id")).alias('song_count')
    )
    .toPandas()
    .T
    .rename(columns={0: "value"})
)

#            limiting    total     percentage
# user_count  298387 / 1019318 =    0.2927
# song_count  81391 /  378310  =    0.2151

print(get_user_counts(triplets_limited).approxQuantile("song_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05))
print(get_song_counts(triplets_limited).approxQuantile("user_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05))

# [3.0, 55.0, 73.0, 129.0, 2523.0]

# [46.0, 79.0, 149.0, 428.0, 44193.0]




# -----------------------------------------------------------------------------
# Encoding
# -----------------------------------------------------------------------------

# Imports

from pyspark.ml.feature import StringIndexer

# Encoding

user_id_indexer = StringIndexer(inputCol="user_id", outputCol="user_id_encoded")
song_id_indexer = StringIndexer(inputCol="song_id", outputCol="song_id_encoded")

user_id_indexer_model = user_id_indexer.fit(triplets_limited)
song_id_indexer_model = song_id_indexer.fit(triplets_limited)

triplets_limited = user_id_indexer_model.transform(triplets_limited)
triplets_limited = song_id_indexer_model.transform(triplets_limited)


# -----------------------------------------------------------------------------
# Splitting
# -----------------------------------------------------------------------------

# Imports

from pyspark.sql.window import *

# Splits

training, test = triplets_limited.randomSplit([0.75, 0.25])

test_not_training = test.join(training, on="user_id", how="left_anti")

training.cache()
test.cache()
test_not_training.cache()

print(f"training:          {training.count()}")
print(f"test:              {test.count()}")
print(f"test_not_training: {test_not_training.count()}")


# training:          20743985
# test:              6910259
# test_not_training: 0


test_not_training.show(50, False)


# +-------+-------+-----+---------------+---------------+
# |user_id|song_id|plays|user_id_encoded|song_id_encoded|
# +-------+-------+-----+---------------+---------------+
# +-------+-------+-----+---------------+---------------+


# Codes dealing with the observation in test but not in training
counts = test_not_training.groupBy("user_id").count().toPandas().set_index("user_id")["count"].to_dict()

temp = (
    test_not_training
    .withColumn("id", monotonically_increasing_id())
    .withColumn("random", rand())
    .withColumn(
        "row",
        row_number()
        .over(
            Window
            .partitionBy("user_id")
            .orderBy("random")
        )
    )
)

for k, v in counts.items():
    temp = temp.where((col("user_id") != k) | (col("row") < v * 0.7))

temp = temp.drop("id", "random", "row")
temp.cache()

temp.show(50, False)


# +-------+-------+-----+---------------+---------------+
# |user_id|song_id|plays|user_id_encoded|song_id_encoded|
# +-------+-------+-----+---------------+---------------+
# +-------+-------+-----+---------------+---------------+


training = training.union(temp.select(training.columns))
test = test.join(temp, on=["user_id", "song_id"], how="left_anti")
test_not_training = test.join(training, on="user_id", how="left_anti")

print(f"training:          {training.count()}")
print(f"test:              {test.count()}")
print(f"test_not_training: {test_not_training.count()}")


# training:          20735231
# test:              6919013
# test_not_training: 0



# -----------------------------------------------------------------------------
# Modeling
# -----------------------------------------------------------------------------

# Imports

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer

from pyspark.mllib.evaluation import RankingMetrics

# Modeling

als = ALS(maxIter=3, regParam=0.01, userCol="user_id_encoded", itemCol="song_id_encoded", ratingCol="plays",implicitPrefs=True)
als_model = als.fit(training)
predictions = als_model.transform(test)

predictions = predictions.orderBy(col("user_id"), col("song_id"), col("prediction").desc())
predictions.cache()

predictions.show(50, False)


# +------------------+----------------------------------------+-----+---------------+---------------+------------+
# |song_id           |user_id                                 |plays|user_id_encoded|song_id_encoded|prediction  |
# +------------------+----------------------------------------+-----+---------------+---------------+------------+
# |SOAYETG12A67ADA751|00007ed2509128dcdd74ea3aac2363e24e9dc06b|2    |230291.0       |668.0          |0.0028589617|
# |SOBWGGV12A6D4FD72E|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |230291.0       |21967.0        |9.727796E-5 |
# |SODESWY12AB0182F2E|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |230291.0       |50140.0        |3.8436243E-5|
# |SOELPFP12A58A7DA4F|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |230291.0       |47791.0        |4.8647635E-5|
# |SOGEWRX12AB0189432|00007ed2509128dcdd74ea3aac2363e24e9dc06b|2    |230291.0       |43771.0        |5.0902414E-5|
# |SOHEWBM12A58A7922A|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |230291.0       |65419.0        |1.1359467E-5|
# |SOHYKCX12A6D4F636F|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |230291.0       |56345.0        |3.0861494E-5|
# |SOICJAD12A8C13B2F4|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |230291.0       |33517.0        |8.227438E-5 |
# |SOINVHR12AB0189418|00007ed2509128dcdd74ea3aac2363e24e9dc06b|2    |230291.0       |53030.0        |4.825835E-5 |
# |SOOKJWB12A6D4FD4F8|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |230291.0       |25135.0        |1.6038015E-4|
# |SOOTFWU12A6D4FB8FB|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |230291.0       |39749.0        |3.0221636E-5|
# |SOPFUBI12A58A79E33|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |230291.0       |70396.0        |1.8131863E-5|
# |SORFZWW12A6D4F742C|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |230291.0       |5890.0         |6.564613E-4 |
# |SOVGNWE12A6D4FB90A|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |230291.0       |50432.0        |2.6998241E-5|
# |SOVMWUC12A8C13750B|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |230291.0       |803.0          |0.005757871 |
# |SOWXPFM12A8C13B2EC|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |230291.0       |44347.0        |3.326081E-5 |
# |SOBKTVL12A8C13C031|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |295314.0       |11850.0        |7.210085E-4 |
# |SOCDAMA12A6D4FB5B6|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |295314.0       |14476.0        |4.3145713E-4|
# |SOCQVSB12A58A80F8B|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |295314.0       |60283.0        |2.313962E-5 |
# |SOEWPBR12A58A79271|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |295314.0       |38082.0        |7.2540286E-5|
# |SOGXDSC12A8C138626|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |295314.0       |41247.0        |1.6320153E-4|
# |SOLLIRG12B35055B4C|00009d93dc719d1dbaf13507725a03b9fdeebebb|2    |295314.0       |7911.0         |3.4796464E-4|
# |SOPEPDC12A6D4F7599|00009d93dc719d1dbaf13507725a03b9fdeebebb|2    |295314.0       |4284.0         |2.1349484E-4|
# |SOQUJSF12A58A7C41A|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |295314.0       |6428.0         |1.99424E-5  |
# |SOVTBQI12A8C142ABA|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |295314.0       |66727.0        |2.3211736E-5|
# |SOZSOOL12AB01834B2|00009d93dc719d1dbaf13507725a03b9fdeebebb|3    |295314.0       |79497.0        |2.703882E-5 |
# |SOCDKAZ12A8C13C80F|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |88217.0        |11684.0        |0.0024955608|
# |SOCDRUZ12A8AE48614|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |88217.0        |6463.0         |0.0052988227|
# |SOCDXHL12A8C137A8A|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |88217.0        |1478.0         |0.011189971 |
# |SODHLYW12A8C135517|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |88217.0        |25856.0        |0.0024787006|
# |SODVYCM12A6D4F7F2A|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|2    |88217.0        |43862.0        |2.2357609E-4|
# |SOEWMIM12A6D4F7982|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |88217.0        |29538.0        |0.0021074007|
# |SOGDDKR12A6701E8FA|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |88217.0        |441.0          |0.0135008255|
# |SOGDTQS12A6310D7D1|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |88217.0        |4306.0         |0.006526065 |
# |SOHTWNJ12A6701D0EB|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|8    |88217.0        |56025.0        |8.0803945E-4|
# |SOIMBGJ12A6D4F828D|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|2    |88217.0        |55792.0        |1.1890371E-4|
# |SOITMVX12AF72A089F|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |88217.0        |6904.0         |0.0019701358|
# |SOIXWDT12A6D4F842F|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |88217.0        |61017.0        |2.4992862E-4|
# |SOJSKXK12A8C13189C|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |88217.0        |52134.0        |1.7291825E-4|
# |SOJXVLM12A6701D0EE|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|9    |88217.0        |18397.0        |0.002174355 |
# |SOKUKJB12A6701C31C|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|2    |88217.0        |11473.0        |0.002474949 |
# |SOMYUZQ12A6D4F7F29|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|2    |88217.0        |61475.0        |8.516592E-5 |
# |SONMHUG12A6D4FB1CF|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |88217.0        |41558.0        |0.001443038 |
# |SOOPYUC12AF72A16C1|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |88217.0        |6315.0         |0.004533518 |
# |SOPJLFV12A6701C797|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |88217.0        |462.0          |0.025597962 |
# |SOPLQZH12A58A78901|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |88217.0        |67887.0        |8.169714E-4 |
# |SORJGVR12A58A7C3C8|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|3    |88217.0        |40302.0        |2.503201E-4 |
# |SORYCTJ12A8C138AC3|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |88217.0        |6704.0         |0.0051517156|
# |SOTLDCX12AAF3B1356|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |88217.0        |2189.0         |0.021566942 |
# |SOTTLBW12A6D4F64AF|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |88217.0        |20158.0        |0.0012070083|
# +------------------+----------------------------------------+-----+---------------+---------------+------------+
# only showing top 50 rows



test.show(5, 100)

# +----------------------------------------+------------------+-----+---------------+---------------+
# |                                 user_id|           song_id|plays|user_id_encoded|song_id_encoded|
# +----------------------------------------+------------------+-----+---------------+---------------+
# |00009d93dc719d1dbaf13507725a03b9fdeebebb|SOVTBQI12A8C142ABA|    1|       295314.0|        66727.0|
# |0000d3c803e068cf1da17724f1674897b2dd7130|SOFPRTU12A58A76BEE|    1|       198283.0|         7657.0|
# |0000f88f8d76a238c251450913b0d070e4a77d19|SOEOIDW12A3F1E9C54|    4|        69197.0|        17325.0|
# |0000f88f8d76a238c251450913b0d070e4a77d19|SOLPTVW12A8C13F136|    2|        69197.0|         6526.0|
# |0000f88f8d76a238c251450913b0d070e4a77d19|SONERDG12AF72A435E|    4|        69197.0|        14236.0|
# +----------------------------------------+------------------+-----+---------------+---------------+
# only showing top 5 rows

hand_selected_userId = ["0000d3c803e068cf1da17724f1674897b2dd7130", "00009d93dc719d1dbaf13507725a03b9fdeebebb", "0000f88f8d76a238c251450913b0d070e4a77d19"]




# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

# Helpers 

def extract_songs_top_k(x, k):
    x = sorted(x, key=lambda x: -x[1])
    return [x[0] for x in x][0:k]

extract_songs_top_k_udf = udf(lambda x: extract_songs_top_k(x, k), ArrayType(IntegerType()))

def extract_songs(x):
    x = sorted(x, key=lambda x: -x[1])
    return [x[0] for x in x]

extract_songs_udf = udf(lambda x: extract_songs(x), ArrayType(IntegerType()))

# Recommendations


# -----------------------------------------------------------------------------
# Evaluate by hand
# -----------------------------------------------------------------------------


k = 5

topK = als_model.recommendForAllUsers(k)

topK.cache()
topK.count()  # 298387

topK.show(10, False)


# +---------------+------------------------------------------------------------------------------------------+
# |user_id_encoded|recommendations                                                                           |
# +---------------+------------------------------------------------------------------------------------------+
# |14             |[[2, 0.7458548], [12, 0.69679046], [0, 0.67386734], [9, 0.66713315], [8, 0.6439969]]      |
# |18             |[[0, 0.85311973], [30, 0.79741216], [7, 0.78468055], [11, 0.73936045], [42, 0.6838447]]   |
# |25             |[[9, 1.219525], [12, 1.1487064], [8, 1.0718989], [2, 0.8999541], [27, 0.8978028]]         |
# |38             |[[12, 0.29205492], [61, 0.2807076], [193, 0.27045736], [19, 0.26912287], [25, 0.26346827]]|
# |46             |[[43, 0.62578523], [7, 0.6026802], [30, 0.5861764], [11, 0.528968], [124, 0.49115923]]    |
# |50             |[[30, 0.485553], [32, 0.43455344], [0, 0.428469], [35, 0.42444128], [48, 0.41261598]]     |
# |73             |[[7, 0.42532456], [11, 0.41611627], [2, 0.4060477], [0, 0.36851645], [39, 0.36210644]]    |
# |97             |[[30, 0.6313347], [0, 0.5629942], [7, 0.5624914], [42, 0.5497694], [88, 0.5419]]          |
# |161            |[[11, 0.6904034], [6, 0.64014685], [88, 0.62964386], [94, 0.62294346], [90, 0.6137737]]   |
# |172            |[[186, 0.3111886], [70, 0.30816197], [2, 0.29265696], [224, 0.28589576], [230, 0.2813927]]|
# +---------------+------------------------------------------------------------------------------------------+
# only showing top 10 rows



recommended_songs = (
    topK
    .withColumn("recommended_songs", extract_songs_top_k_udf(col("recommendations")))
    .select("user_id_encoded", "recommended_songs")
)
recommended_songs.cache()
recommended_songs.count()  # 298387

recommended_songs.show(10, False)


# +---------------+----------------------+
# |user_id_encoded|recommended_songs     |
# +---------------+----------------------+
# |14             |[2, 12, 0, 9, 8]      |
# |18             |[0, 30, 7, 11, 42]    |
# |25             |[9, 12, 8, 2, 27]     |
# |38             |[12, 61, 193, 19, 25] |
# |46             |[43, 7, 30, 11, 124]  |
# |50             |[30, 32, 0, 35, 48]   |
# |73             |[7, 11, 2, 0, 39]     |
# |97             |[30, 0, 7, 42, 88]    |
# |161            |[11, 6, 88, 94, 90]   |
# |172            |[186, 70, 2, 224, 230]|
# +---------------+----------------------+
# only showing top 10 rows



relevant_songs = (
    test
    .select(
        col("user_id_encoded").cast(IntegerType()),
        col("song_id_encoded").cast(IntegerType()),
        col("plays").cast(IntegerType())
    )
    .groupBy('user_id_encoded')
    .agg(
        collect_list(
            array(
                col("song_id_encoded"),
                col("plays")
            )
        ).alias('relevance')
    )
    .withColumn("relevant_songs", extract_songs_udf(col("relevance")))
    .select("user_id_encoded", "relevant_songs")
)
relevant_songs.cache()
relevant_songs.count() # 298373

relevant_songs.show(10, False)

# +---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |user_id_encoded|relevant_songs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
# +---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |14             |[8958, 1253, 41647, 51149, 48312, 39234, 88, 11003, 11321, 3701, 11464, 36159, 73110, 62906, 14207, 4471, 15096, 13685, 22512, 35070, 9259, 49657, 25586, 4329, 7, 13533, 2443, 64, 8619, 11268, 27709, 4877, 23827, 17900, 61325, 10362, 12092, 17449, 25119, 15981, 14487, 16571, 9599, 11477, 5122, 12339, 5582, 11250, 11545, 1565, 13837, 63288, 4496, 1008, 58363, 20580, 25315, 274, 15411, 110, 41797, 435, 6265, 5097, 5736, 45785, 20995, 39360, 24194, 6604, 7389, 62342, 17633, 48992, 2840, 689, 43569, 1927, 5800, 43492, 69589, 8840, 17485, 19064, 58225, 261, 62565, 2744, 657, 28868, 11957, 1505, 42046, 47253, 38871, 5994, 498, 59306, 6758, 33376, 3849, 6744, 2282, 5872, 15145, 423, 37195, 26595, 16698, 237, 5544, 3895, 22906, 27097, 1013, 35572, 11279, 7123, 4368, 4426, 76208, 16626, 28336, 29635, 4407, 53979, 18437, 38800, 518, 11117, 12007, 31875, 19200, 1659, 40959, 864, 11371, 1838, 13138, 7742, 36987, 15106, 16712, 538, 3113, 6956, 3769, 66601, 77993, 27113, 26654, 346, 15001, 3343, 920, 1023, 9826, 3850, 12366, 28789, 39751, 238, 29634, 19362, 18481, 13102, 6129, 60662, 7819, 942, 24295, 11406, 15061, 14790, 26816, 12289, 27913, 7654, 41369, 1048, 15222, 10165, 2988, 5772, 24078, 36070, 4794, 4479, 15005, 67, 3393, 19726, 6193, 14084, 2255, 2456, 7338, 29443, 47947, 47856, 132, 2937, 3118, 16385, 24733, 8607, 12831, 1403, 9344, 18759, 58966, 7176, 31269, 6047, 1196, 23280, 164, 15510, 7282, 2515, 23345, 15498, 1309, 7351, 5487, 9462, 1715, 24142, 11686, 8490, 14066, 53729, 4762, 226, 1819, 4490, 5612, 9652, 9210, 12437, 14571, 3903, 3425, 12783, 34226, 22509, 23577, 19569, 7831, 5315, 2114, 6795, 7792, 81294, 2681, 14634, 29643, 30253, 42982, 27163, 3836, 8907, 10371, 43449, 2065, 35503, 3422, 1520, 65077, 26827, 27482, 1019, 16665, 9987, 459, 2238, 241, 6000, 781, 163, 9380, 2960, 55846]|
# |18             |[57748, 8272, 75246, 70502, 15817, 1414, 43842, 62020, 48978, 8019, 7129, 41392, 22807, 10116, 78850, 40107, 1327, 13190, 367, 2418, 20684, 21996, 43028, 23306, 65416, 26816, 30769, 47057, 79207, 33795, 12959, 1567, 16592, 80337, 11734, 375, 13181, 2086, 14990, 13, 11521, 233, 348, 1339, 31280, 13622, 47555, 44572, 22356, 5486, 110, 14443, 11183, 24078, 7298, 189, 21664, 148, 8800, 3387, 48457, 12846, 2791, 333, 27304, 23018, 28983, 6300, 2908, 25513, 20379, 31513, 4911, 363, 4781, 47305, 255, 12396, 554, 6289, 39411, 35032, 20909, 16, 37062, 29282, 42537, 6897, 58805, 13353, 845, 1877, 3390, 14839, 16703, 18793, 47205, 29508, 2367, 27818, 3895, 15384, 2942, 11987, 11078, 29683, 29685, 16441, 185, 2007, 14371, 10854, 44669, 3183, 5104, 20251, 759, 9925, 26, 14011, 448, 39232, 2464, 6779, 22564, 4398, 14689, 43154, 897, 66792, 16073, 25858, 1066, 569, 15873, 396, 41931, 13324, 24036, 8367, 7787, 1665, 3253, 790, 836, 3416, 55843, 183, 30177, 21187, 2793, 8854, 4796, 9745, 5189, 3395, 2554, 804, 644, 39607, 1080, 3704, 32178, 70154, 19909, 1609, 76278, 65822, 273, 3922, 31596, 4682, 10988, 1654, 62004, 6074, 1201, 6294, 524, 50669, 840, 15771, 2777, 13699, 403, 56116, 14026, 2369, 58306, 417, 56411, 66035, 1768, 13107, 4669, 47863, 1818, 23984, 15805, 3185, 2227, 9683, 567, 13034, 2948, 71024, 68023, 279, 22866, 3731, 1486, 62610, 2293, 11322, 4351, 1990, 32443, 16031, 28857, 8075, 56630, 523, 22609, 8, 2270, 72112, 78036, 537, 19903, 29283, 79300, 58299, 19339, 15089, 18317, 15582, 3141, 12622, 3239, 10122, 3705, 51274, 9230, 39445, 52410, 2678, 58384, 25432, 2134, 72700, 1177, 12032, 8715, 968, 20898, 4453, 6174, 18049, 40560, 18392, 1547, 1107, 36961]                                                                                                                                           |
# |25             |[6096, 5125, 19659, 5985, 7682, 1520, 8357, 2369, 191, 3061, 329, 7163, 18001, 170, 8811, 2312, 34811, 207, 24, 1593, 642, 572, 471, 1981, 211, 1052, 1864, 50, 1326, 1356, 44, 2559, 611, 25313, 2681, 481, 42185, 52071, 1477, 3841, 229, 14461, 18844, 5088, 692, 2490, 8746, 426, 14165, 7543, 11450, 381, 445, 1437, 2655, 3696, 7149, 1884, 5063, 520, 1535, 1480, 733, 698, 1691, 990, 20054, 113, 5580, 2128, 644, 1224, 741, 13898, 1443, 9470, 2519, 1137, 3967, 925, 1837, 365, 2427, 37862, 64478, 8307, 13135, 214, 1320, 5510, 4080, 1227, 2350, 22392, 2017, 510, 9585, 4867, 2384, 2928, 727, 10367, 557, 1289, 596, 812, 50256, 64709, 519, 239, 1233, 1769, 2944, 27927, 2095, 21510, 3016, 2104, 1702, 11104, 2060, 3542, 6469, 8870, 12163, 4538, 2573, 479, 477, 13908, 858, 22512, 13991, 4292, 3518, 8060, 6345, 1309, 10164, 10384, 1862, 6838, 1087, 636, 4314, 1184, 21205, 34, 32406, 713, 464, 3159, 1698, 11491, 32480, 681, 951, 308, 3053, 756, 8613, 5592, 696, 120, 14176, 8391, 6176, 2252, 1250, 28716, 4542, 20863, 215, 25284, 3097, 796, 47734, 1247, 1230, 7748, 65454, 3720, 10215, 6293, 171, 2971, 10495, 901, 2722, 907, 2466, 179, 1077, 348, 7917, 40550, 6412, 243, 936, 3348, 6923, 34263, 12614, 31768, 3913, 7724, 6116, 55788, 12590, 57304, 27762, 43499, 4603, 1631, 1980, 6298, 14265, 2374, 12518, 1646, 2446, 543, 1354, 3600, 74885, 4153, 4877, 53728, 3736, 8614, 331, 40292, 5429, 4555, 6770, 3687, 7406, 1604, 3044, 6362, 49977, 2325, 22154, 25883, 757, 4289, 27174, 2064, 2754, 6996, 37527, 24455, 22029, 1405, 15426, 17991, 68077, 849, 49320, 3672, 4689, 587, 40154, 48818, 241, 23438, 893, 764, 22895, 6452, 44293, 9736, 51603, 1398, 21452, 563, 1723, 3218, 570, 9639, 2311, 1749, 4673, 71936, 48113, 161, 2150, 9730, 506, 2330, 3202, 80042, 700, 22871, 21919, 31583, 3252, 3296, 22889]                    |
# |38             |[236, 8320, 4270, 20340, 32416, 50142, 30716, 15550, 273, 62705, 30737, 55730, 3377, 416, 4155, 13130, 8699, 325, 71654, 20951, 16923, 5986, 14393, 17821, 4986, 18580, 11949, 14694, 1521, 8521, 6881, 2732, 36357, 17384, 4524, 5824, 11189, 1584, 10319, 18145, 218, 14592, 6079, 17123, 33668, 350, 443, 60704, 8817, 14187, 13657, 5292, 45347, 5778, 22478, 36288, 20593, 9386, 4247, 23534, 13363, 33347, 11879, 14048, 28739, 7720, 606, 7750, 2395, 19145, 4180, 38339, 16514, 3391, 31732, 2688, 6516, 4668, 2867, 3320, 9501, 5930, 4855, 27243, 24561, 190, 6375, 11208, 30685, 29229, 11329, 38143, 20925, 19109, 19279, 37596, 128, 23007, 2865, 20666, 61967, 16761, 29645, 19890, 10764, 33409, 7868, 19633, 19684, 13999, 3810, 17081, 19052, 9976, 27994, 6944, 33712, 9705, 8911, 26679, 25489, 39637, 30314, 4056, 39531, 12765, 32855, 9297, 20355, 34482, 17160, 52705, 11322, 41790, 28207, 40891, 13590, 12574, 45500, 44672, 3375, 25452, 22510, 996, 40788, 15623, 43499, 32299, 1357, 36592, 77392, 29215, 33389, 30074, 11751, 21176, 48653, 3782, 4480, 13293, 11180, 12683, 14761, 3426, 41668, 2731, 20084, 42124, 4947, 2294, 36526, 11603, 46801, 4108, 29980, 11146, 19113, 16472, 16515, 33863, 45797, 9324, 27770, 3153, 13309, 38705, 52384, 8187, 23797, 6925, 37607, 20407, 38323, 4736, 9543, 17260, 5445, 9394, 11111, 4005, 29522, 2981, 16331, 16217, 67671, 1790, 12784, 20787, 12862, 27535, 20387, 12903, 10910, 14174, 72753, 19288, 2284, 48665, 35251, 4738, 40261, 10233, 1878, 5189, 64296, 7494, 52001, 31774, 38426, 11222, 34553, 29020, 27866, 11927, 20698, 17632]                                                                                                                                                                                                                                                                 |
# |46             |[41486, 16109, 8205, 5119, 955, 15442, 16889, 3216, 8291, 6153, 27001, 15890, 241, 1224, 1828, 22429, 179, 44082, 21213, 3230, 50866, 44066, 24262, 53404, 4041, 48423, 16334, 7874, 34105, 31572, 35007, 14767, 9774, 50652, 65353, 53161, 52983, 23931, 36520, 13461, 13026, 29351, 77360, 69694, 54080, 910, 15902, 2456, 1791, 1810, 22223, 189, 2157, 74036, 12502, 77004, 3396, 36137, 273, 15124, 3052, 6093, 26978, 811, 2137, 2972, 29946, 63338, 62557, 29145, 2155, 769, 6092, 4440, 588, 652, 8935, 7089, 53694, 56851, 76005, 3064, 19180, 11876, 20255, 14723, 11811, 40172, 6289, 7890, 204, 9223, 8650, 50230, 52115, 802, 45518, 70629, 7097, 7006, 17865, 1787, 1293, 248, 14698, 12653, 67010, 57079, 53935, 1252, 21298, 18433, 13400, 49698, 16148, 18989, 2756, 64901, 68639, 1324, 13652, 20119, 6263, 33550, 14296, 14531, 68642, 28667, 18294, 56579, 489, 19696, 1262, 1489, 19874, 440, 4937, 3544, 12273, 55040, 25735, 47454, 6375, 9847, 2106, 26998, 5344, 30845, 1727, 29938, 81196, 76187, 13297, 2274, 18532, 8845, 46245, 2180, 1499, 8246, 7246, 55803, 59, 52448, 800, 20967, 13351, 2429, 7623, 8348, 5129, 419, 15384, 23166, 403, 21903, 21890, 42433, 63576, 2985, 3573, 24317, 19366, 24254, 36674, 1022, 23772, 63518, 700, 1585, 363, 1627, 54995, 6014, 649, 9231, 2076, 27666, 1961, 50062, 1377, 66354, 4513, 4867, 29646, 13380, 1836, 17112, 45873, 29449, 12940, 280, 6458]                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
# |50             |[1026, 1265, 4891, 5035, 1872, 9355, 25224, 8115, 9509, 15214, 21091, 19515, 22427, 18621, 4906, 5966, 17010, 3290, 7601, 279, 23741, 15359, 358, 248, 15142, 3017, 12405, 455, 15409, 5951, 9655, 20396, 27122, 15791, 29569, 3953, 30676, 7592, 4029, 15213, 14812, 11177, 8837, 11280, 36015, 9677, 3963, 6897, 10969, 11231, 857, 2487, 15839, 22998, 14512, 21973, 14211, 4403, 41433, 11282, 16197, 10398, 10439, 4607, 4453, 32879, 22095, 15842, 8957, 7501, 64110, 5883, 620, 13751, 29159, 6937, 27569, 8038, 1190, 4827, 4389, 4522, 9678, 19810, 27166, 31602, 4287, 25579, 79495, 21872, 50206, 4418, 27836, 1919, 59181, 3799, 381, 2936, 1926, 10132, 22629, 9904, 15849, 6031, 13054, 573, 17528, 29615, 52209, 56484, 5847, 69, 43284, 64237, 8035, 20799, 10834, 3818, 11343, 39205, 2998, 51773, 5477, 12893, 9730, 10221, 1674, 8854, 16356, 3352, 8778, 5639, 9514, 19284, 14701, 555, 7692, 17787, 7785, 37703, 8752, 22723, 26257, 14483, 29912, 415, 48831, 1583, 31233, 14185, 514, 43881, 10476, 14847, 3336, 17641, 30516, 5076, 590, 12376, 19391, 3395, 41, 16812, 40217, 8769, 9689, 276, 13320, 400, 1982, 2749, 10567, 4519, 10510, 2767, 1818, 15135, 29340, 21258, 843, 14948, 23496, 30189, 7891, 13610, 13527, 16184, 20811, 10847, 3666, 787, 30885, 7607, 37061, 23552, 249, 31002, 3343, 44552, 15073, 6136, 15105, 27321, 12246, 10512, 7373, 7969, 3146, 6398, 26764, 18262, 28621, 13022, 16848, 21039, 671, 18574, 4015, 29886, 1526, 6058, 42256]                                                                                                                                                                                                                                                                                                                                                                                              |
# |73             |[7737, 21868, 24198, 8538, 13286, 8135, 544, 43178, 9099, 8079, 44350, 26061, 453, 4942, 37111, 3970, 146, 627, 3034, 17837, 2981, 1267, 821, 262, 17590, 1713, 9910, 32600, 8823, 48417, 11575, 7310, 629, 3189, 11888, 8783, 20943, 52789, 15214, 54284, 54510, 33146, 56802, 33959, 8992, 18428, 3205, 8877, 37735, 51989, 69425, 25437, 1258, 23010, 27938, 10121, 35994, 3163, 7185, 11905, 12756, 1355, 37666, 4843, 24138, 19391, 13620, 7043, 26209, 35394, 15663, 1633, 33710, 878, 13013, 2641, 39227, 50529, 5402, 2113, 7407, 1050, 16022, 3314, 1457, 9307, 443, 1931, 12856, 41573, 6042, 2404, 2990, 5157, 32414, 3394, 9428, 4965, 18189, 1164, 14113, 15490, 80846, 5277, 3128, 38449, 319, 35825, 3260, 2242, 9450, 24544, 10246, 1260, 695, 57219, 27064, 4041, 11524, 12796, 7061, 13828, 680, 5778, 10226, 75458, 4985, 180, 23627, 18996, 7716, 8886, 1750, 55550, 18432, 24254, 35791, 2316, 24446, 8296, 26200, 14285, 32617, 25140, 18747, 14738, 1125, 2604, 40135, 19871, 13529, 40425, 18388, 24804, 5691, 24633, 10372, 16618, 48648, 12430, 9170, 16230, 40261, 2999, 5508, 4279, 31359, 18065, 37110, 9689, 28907, 4116, 7157, 1705, 641, 41205, 2828, 2599, 28058, 25681, 432, 2008, 32080, 13033, 33102, 1308, 6919, 5479, 41575, 33188, 33918, 16750, 24272, 1998, 18816, 5869, 26358, 9066, 56268, 16032, 12365, 15221, 2417]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
# |97             |[10970, 27174, 18223, 13417, 17120, 13327, 52123, 38963, 32585, 37319, 7104, 22175, 12517, 19652, 16850, 38703, 29807, 26506, 10610, 65221, 36950, 63492, 8449, 10598, 3100, 52243, 11058, 54877, 33119, 70486, 9880, 25555, 40630, 574, 8841, 33288, 38823, 233, 8055, 44000, 36032, 3948, 1458, 60997, 356, 41604, 23844, 5421, 762, 351, 25478, 6887, 4026, 3040, 1877, 14681, 7541, 7339, 10643, 31016, 3271, 48972, 13378, 30385, 4713, 2907, 25381, 56283, 7675, 2972, 21701, 150, 4489, 59697, 4053, 859, 19501, 24466, 1436, 30932, 1828, 12907, 55715, 88, 1622, 30043, 55755, 18926, 9922, 5397, 2028, 51624, 2218, 3534, 1474, 5121, 280, 49347, 531, 17891, 8005, 43263, 45100, 13, 1807, 270, 4094, 25714, 6107, 6446, 1556, 21968, 220, 18393, 77122, 72606, 64001, 28157, 311, 40867, 18747, 898, 5149, 45760, 6009, 20874, 980, 6775, 10138, 23668, 27667, 29757, 45340, 2239, 5473, 79328, 218, 3354, 76372, 5894, 40872, 124, 2354, 13843, 17280, 27488, 47753, 77498, 17666, 13033, 579, 7884, 30989, 4027, 24856, 4394, 60905, 16639, 11234, 48616, 60175, 3866, 28396, 90, 1100, 67278, 60144, 4145, 4102, 41602, 18045, 10294, 14091, 1256, 40878, 374, 17773, 6170, 9663, 7074, 25745, 5592, 2184, 2119, 23110, 21182, 20111, 8082, 77944, 28853, 24170, 14440, 42618, 6075, 13419, 3603, 3598, 3552, 45980, 6814, 66788, 3618, 69797, 393, 10128, 5533, 32698, 56980, 66385, 46259, 628, 20215, 2809, 12002]                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
# |161            |[8077, 5221, 5499, 4230, 684, 15326, 385, 4981, 7807, 14976, 18409, 9248, 4709, 11228, 5269, 48058, 49319, 79909, 3754, 392, 3741, 14595, 5043, 9855, 251, 3970, 10082, 6818, 18894, 7452, 191, 2965, 10725, 17939, 2397, 11874, 2499, 16056, 5995, 4093, 422, 1388, 8019, 4690, 7241, 3615, 10815, 5902, 10493, 4853, 53050, 1326, 426, 64805, 3849, 1548, 28897, 4590, 2226, 2402, 6823, 15045, 21246, 10668, 3702, 5944, 5167, 36477, 29635, 6194, 681, 3471, 338, 563, 63082, 15086, 61099, 31964, 200, 8907, 20830, 70498, 11028, 14511, 543, 508, 2191, 2802, 15223, 13644, 22862, 10308, 10684, 8910, 28577, 45068, 1544, 8096, 2350, 65887, 11500, 78926, 2673, 1777, 38494, 5757, 89, 17121, 5229, 28327, 740, 8262, 12713, 8771, 36155, 9769, 25404, 12102, 21409, 14056, 14613, 6401, 11810, 15220, 16480, 16428, 16680, 4890, 21927, 37638, 14220, 14036, 14992, 6387, 7888, 34128, 1475, 46135, 41722, 21484, 2602, 7921, 26657, 10107, 1845, 47543, 1507, 16427, 434, 8340, 17806, 3712, 37369, 6605, 567, 3890, 6873, 43615, 5982, 5136, 53942, 34650, 5802, 5879, 25593, 6970, 51305, 595, 8712, 41402, 12548, 1255, 1936, 16050, 16199, 8162]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
# |172            |[9517, 5158, 11096, 9899, 13472, 14750, 17000, 20977, 21335, 24721, 49892, 35672, 11462, 20082, 3739, 14419, 26315, 10448, 19712, 7105, 14561, 18953, 33567, 45890, 11415, 4594, 1463, 9986, 15340, 44544, 13372, 14368, 19228, 23008, 3436, 18066, 35054, 13055, 7399, 7580, 8028, 13316, 10126, 709, 23161, 1067, 31321, 9913, 66070, 11414, 24105, 2406, 67410, 24282, 3167, 1499, 19684, 21178, 203, 8756, 53516, 75259, 29846, 22778, 383, 4766, 36782, 52605, 32441, 845, 13174, 41886, 2530, 5990, 878, 16290, 1157, 265, 22829, 6777, 47773, 31618, 21428, 19038, 6888, 11906, 40321, 1477, 4541, 13424, 58293, 64666, 10915, 39045, 76490, 716, 18173, 66391, 2597, 27025, 2214, 7359, 7598, 19164, 10694, 71102, 13586, 10351, 14230, 11586, 11891, 536, 18257, 1095, 14234, 39861, 36704, 6669, 7918, 7198, 25941, 8356, 33726, 41350, 3385, 48787, 13366, 22457, 35719, 55282, 6694, 9991, 31808, 1657, 1378, 26092, 516, 18465, 14143, 23564, 37676, 934, 8786, 35011, 20888, 1750, 5778, 42747, 43035, 30422, 28860, 7179, 1050, 32998, 7808, 28548, 32403, 2628, 14523, 30955, 28553, 1321, 31834, 24922, 50007, 5653, 19478, 14634, 66950, 22548, 36205, 6166, 19012, 42963, 52, 30163, 22764, 36387, 6136, 15350, 3748, 45358, 7884, 31442, 8684, 43306, 36696, 23713, 33562, 1665, 3361]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
# +---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# only showing top 10 rows



combined = (
    recommended_songs.join(relevant_songs, on='user_id_encoded', how='inner')
    .rdd
    .map(lambda row: (row[1], row[2]))
)
combined.cache()
combined.count()  #  298373


combined.take(1)


# [([2, 12, 0, 9, 8],
  # [8958,
   # 1253,
   # 41647,
   # 51149,
   # 48312,
   # 39234,
   # 88,
   # 11003,
   # 11321,
   # 3701,
   # 11464,
   # 36159,
   # 73110,
   # 62906,
   # 14207,
   # 4471,
   # 15096,
   # 13685,
   # 22512,
   # 35070,
   # 9259,
   # 49657,
   # 25586,
   # 4329,
   # 7,
   # 13533,
   # 2443,
   # 64,
   # 8619,
   # 11268,
   # 27709,
   # 4877,
   # 23827,
   # 17900,
   # 61325,
   # 10362,
   # 12092,
   # 17449,
   # 25119,
   # 15981,
   # 14487,
   # 16571,
   # 9599,
   # 11477,
   # 5122,
   # 12339,
   # 5582,
   # 11250,
   # 11545,
   # 1565,
   # 13837,
   # 63288,
   # 4496,
   # 1008,
   # 58363,
   # 20580,
   # 25315,
   # 274,
   # 15411,
   # 110,
   # 41797,
   # 435,
   # 6265,
   # 5097,
   # 5736,
   # 45785,
   # 20995,
   # 39360,
   # 24194,
   # 6604,
   # 7389,
   # 62342,
   # 17633,
   # 48992,
   # 2840,
   # 689,
   # 43569,
   # 1927,
   # 5800,
   # 43492,
   # 69589,
   # 8840,
   # 17485,
   # 19064,
   # 58225,
   # 261,
   # 62565,
   # 2744,
   # 657,
   # 28868,
   # 11957,
   # 1505,
   # 42046,
   # 47253,
   # 38871,
   # 5994,
   # 498,
   # 59306,
   # 6758,
   # 33376,
   # 3849,
   # 6744,
   # 2282,
   # 5872,
   # 15145,
   # 423,
   # 37195,
   # 26595,
   # 16698,
   # 237,
   # 5544,
   # 3895,
   # 22906,
   # 27097,
   # 1013,
   # 35572,
   # 11279,
   # 7123,
   # 4368,
   # 4426,
   # 76208,
   # 16626,
   # 28336,
   # 29635,
   # 4407,
   # 53979,
   # 18437,
   # 38800,
   # 518,
   # 11117,
   # 12007,
   # 31875,
   # 19200,
   # 1659,
   # 40959,
   # 864,
   # 11371,
   # 1838,
   # 13138,
   # 7742,
   # 36987,
   # 15106,
   # 16712,
   # 538,
   # 3113,
   # 6956,
   # 3769,
   # 66601,
   # 77993,
   # 27113,
   # 26654,
   # 346,
   # 15001,
   # 3343,
   # 920,
   # 1023,
   # 9826,
   # 3850,
   # 12366,
   # 28789,
   # 39751,
   # 238,
   # 29634,
   # 19362,
   # 18481,
   # 13102,
   # 6129,
   # 60662,
   # 7819,
   # 942,
   # 24295,
   # 11406,
   # 15061,
   # 14790,
   # 26816,
   # 12289,
   # 27913,
   # 7654,
   # 41369,
   # 1048,
   # 15222,
   # 10165,
   # 2988,
   # 5772,
   # 24078,
   # 36070,
   # 4794,
   # 4479,
   # 15005,
   # 67,
   # 3393,
   # 19726,
   # 6193,
   # 14084,
   # 2255,
   # 2456,
   # 7338,
   # 29443,
   # 47947,
   # 47856,
   # 132,
   # 2937,
   # 3118,
   # 16385,
   # 24733,
   # 8607,
   # 12831,
   # 1403,
   # 9344,
   # 18759,
   # 58966,
   # 7176,
   # 31269,
   # 6047,
   # 1196,
   # 23280,
   # 164,
   # 15510,
   # 7282,
   # 2515,
   # 23345,
   # 15498,
   # 1309,
   # 7351,
   # 5487,
   # 9462,
   # 1715,
   # 24142,
   # 11686,
   # 8490,
   # 14066,
   # 53729,
   # 4762,
   # 226,
   # 1819,
   # 4490,
   # 5612,
   # 9652,
   # 9210,
   # 12437,
   # 14571,
   # 3903,
   # 3425,
   # 12783,
   # 34226,
   # 22509,
   # 23577,
   # 19569,
   # 7831,
   # 5315,
   # 2114,
   # 6795,
   # 7792,
   # 81294,
   # 2681,
   # 14634,
   # 29643,
   # 30253,
   # 42982,
   # 27163,
   # 3836,
   # 8907,
   # 10371,
   # 43449,
   # 2065,
   # 35503,
   # 3422,
   # 1520,
   # 65077,
   # 26827,
   # 27482,
   # 1019,
   # 16665,
   # 9987,
   # 459,
   # 2238,
   # 241,
   # 6000,
   # 781,
   # 163,
   # 9380,
   # 2960,
   # 55846])]



# -----------------------------------------------------------------------------
# Metrics  & k = 5 (Precision @5) 
# -----------------------------------------------------------------------------

# k = 5

rankingMetrics = RankingMetrics(combined)
precisionAtK = rankingMetrics.precisionAt(k)
print(precisionAtK)  # 0.048997047280380984

# -----------------------------------------------------------------------------
# Metrics  & k = 5 (meanAveragePrecision) 
# -----------------------------------------------------------------------------


rankingMetrics = RankingMetrics(combined)
meanAveragePrecision = rankingMetrics.meanAveragePrecision
print(meanAveragePrecision) # 0.006350660688257587


# -----------------------------------------------------------------------------
# Metrics  & k = 10 (NDCG @ 10) 
# -----------------------------------------------------------------------------
# Recommendations

k = 10

topK = als_model.recommendForAllUsers(k)

topK.cache()
topK.count()  # 298387

topK.show(10, False)

# +---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |user_id_encoded|recommendations                                                                                                                                                                       |
# +---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |14             |[[2, 0.7458548], [12, 0.69679046], [0, 0.67386734], [9, 0.66713315], [8, 0.6439969], [7, 0.6266936], [11, 0.59886223], [24, 0.58883023], [15, 0.5702793], [48, 0.5678326]]            |
# |18             |[[0, 0.85311973], [30, 0.79741216], [7, 0.78468055], [11, 0.73936045], [42, 0.6838447], [43, 0.6430503], [81, 0.63092893], [15, 0.62890786], [31, 0.62874], [49, 0.5915952]]          |
# |25             |[[9, 1.219525], [12, 1.1487064], [8, 1.0718989], [2, 0.8999541], [27, 0.8978028], [96, 0.88224965], [24, 0.83935046], [19, 0.8341485], [5, 0.7760895], [13, 0.7292934]]               |
# |38             |[[12, 0.29205492], [61, 0.2807076], [193, 0.27045736], [19, 0.26912287], [25, 0.26346827], [289, 0.25659344], [164, 0.2523878], [29, 0.24752165], [9, 0.24664289], [236, 0.24421258]] |
# |46             |[[43, 0.62578523], [7, 0.6026802], [30, 0.5861764], [11, 0.528968], [124, 0.49115923], [48, 0.4644532], [197, 0.4575749], [88, 0.44325367], [72, 0.43835273], [89, 0.43601635]]       |
# |50             |[[30, 0.485553], [32, 0.43455344], [0, 0.428469], [35, 0.42444128], [48, 0.41261598], [7, 0.41059667], [43, 0.40546966], [12, 0.40543485], [76, 0.3897429], [38, 0.3816371]]          |
# |73             |[[7, 0.42532456], [11, 0.41611627], [2, 0.4060477], [0, 0.36851645], [39, 0.36210644], [15, 0.3608787], [64, 0.35998893], [30, 0.35566], [48, 0.34837478], [43, 0.34362927]]          |
# |97             |[[30, 0.6313347], [0, 0.5629942], [7, 0.5624914], [42, 0.5497694], [88, 0.5419], [11, 0.5377039], [81, 0.519968], [76, 0.49553436], [124, 0.45654932], [31, 0.45551836]]              |
# |161            |[[11, 0.6904034], [6, 0.64014685], [88, 0.62964386], [94, 0.62294346], [90, 0.6137737], [7, 0.5913049], [2, 0.5889416], [12, 0.56179756], [71, 0.5535238], [89, 0.54847056]]          |
# |172            |[[186, 0.3111886], [70, 0.30816197], [2, 0.29265696], [224, 0.28589576], [230, 0.2813927], [188, 0.27762726], [212, 0.2762976], [48, 0.26511425], [234, 0.26131755], [11, 0.25989822]]|
# +---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# only showing top 10 rows



recommended_songs = (
    topK
    .withColumn("recommended_songs", extract_songs_top_k_udf(col("recommendations")))
    .select("user_id_encoded", "recommended_songs")
)
recommended_songs.cache()
recommended_songs.count()  # 298387

recommended_songs.show(10, False)


""" There is something wrong with the code, only generate 5 not 10.
"""
# +---------------+----------------------+
# |user_id_encoded|recommended_songs     |
# +---------------+----------------------+
# |14             |[2, 12, 0, 9, 8]      |
# |18             |[0, 30, 7, 11, 42]    |
# |25             |[9, 12, 8, 2, 27]     |
# |38             |[12, 61, 193, 19, 25] |
# |46             |[43, 7, 30, 11, 124]  |
# |50             |[30, 32, 0, 35, 48]   |
# |73             |[7, 11, 2, 0, 39]     |
# |97             |[30, 0, 7, 42, 88]    |
# |161            |[11, 6, 88, 94, 90]   |
# |172            |[186, 70, 2, 224, 230]|
# +---------------+----------------------+
# only showing top 10 rows

"""  when k = 5 below, totally different, why?
# +---------------+-----------------------+
# |user_id_encoded|recommended_songs      |
# +---------------+-----------------------+
# |14             |[11, 7, 0, 2, 5]       |
# |18             |[0, 11, 42, 7, 31]     |
# |25             |[9, 12, 8, 24, 27]     |
# |38             |[359, 13, 12, 85, 61]  |
# |46             |[0, 11, 7, 42, 43]     |
# |50             |[43, 48, 30, 0, 12]    |
# |73             |[64, 39, 30, 11, 48]   |
# |97             |[0, 11, 42, 81, 13]    |
# |161            |[2, 90, 11, 94, 9]     |
# |172            |[70, 186, 39, 188, 212]|
# +---------------+-----------------------+
# only showing top 10 rows

"""
relevant_songs = (
    test
    .select(
        col("user_id_encoded").cast(IntegerType()),
        col("song_id_encoded").cast(IntegerType()),
        col("plays").cast(IntegerType())
    )
    .groupBy('user_id_encoded')
    .agg(
        collect_list(
            array(
                col("song_id_encoded"),
                col("plays")
            )
        ).alias('relevance')
    )
    .withColumn("relevant_songs", extract_songs_udf(col("relevance")))
    .select("user_id_encoded", "relevant_songs")
)
relevant_songs.cache()
relevant_songs.count() # 298369

relevant_songs.show(10, False)


# +---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |user_id_encoded|relevant_songs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
# +---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |14             |[8958, 41647, 1253, 48312, 51149, 39234, 11321, 88, 11003, 3701, 14207, 62906, 11464, 36159, 73110, 49657, 2443, 25586, 4329, 7, 4471, 15096, 22512, 35070, 9259, 13533, 13685, 17900, 11268, 27709, 4877, 8619, 64, 23827, 11477, 4496, 1008, 12339, 5582, 11250, 9599, 5122, 10362, 12092, 17449, 25119, 15981, 14487, 16571, 11545, 1565, 13837, 63288, 61325, 6604, 7389, 62342, 17633, 48992, 47253, 38871, 5994, 498, 59306, 8840, 17485, 19064, 58225, 261, 62565, 2744, 657, 58363, 20580, 2840, 689, 43569, 1927, 5800, 45785, 20995, 39360, 24194, 6265, 5097, 5736, 43492, 69589, 28868, 11957, 1505, 42046, 25315, 274, 15411, 110, 41797, 435, 942, 24295, 11406, 15061, 14790, 26816, 12289, 27913, 7654, 41369, 1048, 15222, 10165, 2988, 5772, 24078, 36070, 4794, 2065, 35503, 3422, 1520, 65077, 26827, 27482, 1019, 16665, 9987, 459, 2238, 241, 6000, 781, 163, 9380, 2960, 55846, 15510, 7282, 2515, 23345, 15498, 1309, 7351, 5487, 9462, 1715, 24142, 11686, 8490, 14066, 53729, 4762, 226, 1819, 4490, 5612, 9652, 9210, 12437, 14571, 3903, 3425, 12783, 34226, 22509, 23577, 19569, 7831, 5315, 2114, 6795, 6758, 33376, 3849, 6744, 2282, 5872, 15145, 423, 37195, 26595, 16698, 237, 5544, 3895, 22906, 27097, 1013, 35572, 4479, 15005, 67, 3393, 19726, 6193, 14084, 2255, 2456, 7338, 29443, 19362, 18481, 13102, 6129, 60662, 7819, 7742, 36987, 15106, 16712, 538, 3113, 6956, 3769, 66601, 77993, 27113, 26654, 346, 15001, 3343, 47947, 47856, 132, 2937, 3118, 16385, 24733, 8607, 12831, 1403, 9344, 18759, 58966, 7176, 31269, 6047, 1196, 23280, 164, 920, 1023, 9826, 3850, 12366, 28789, 39751, 238, 29634, 7792, 81294, 2681, 14634, 29643, 30253, 42982, 27163, 3836, 8907, 10371, 43449, 11279, 7123, 4368, 4426, 76208, 16626, 28336, 29635, 4407, 53979, 18437, 38800, 518, 11117, 12007, 31875, 19200, 1659, 40959, 864, 11371, 1838, 13138]|
# |18             |[57748, 8272, 75246, 1414, 15817, 70502, 43842, 62020, 48978, 41392, 8019, 7129, 78850, 22807, 10116, 40107, 13190, 1327, 23306, 12959, 30769, 47057, 367, 2418, 20684, 21996, 65416, 26816, 43028, 79207, 33795, 80337, 11734, 1567, 16592, 375, 44572, 22356, 21664, 14443, 11183, 24078, 13181, 2086, 14990, 5486, 31280, 13622, 47555, 11521, 233, 348, 110, 1339, 7298, 189, 13, 554, 3390, 14839, 37062, 29282, 42537, 6897, 58805, 13353, 148, 8800, 3387, 48457, 12846, 2791, 333, 27304, 6289, 39411, 363, 4781, 47305, 255, 12396, 2908, 25513, 20379, 31513, 35032, 20909, 16, 4911, 845, 1877, 23018, 28983, 6300, 9745, 5189, 3395, 2554, 804, 644, 39607, 1080, 3704, 32178, 70154, 19909, 1609, 76278, 65822, 968, 20898, 4453, 6174, 18049, 40560, 18392, 1547, 1107, 36961, 4669, 47863, 1818, 23984, 15805, 3185, 2227, 9683, 567, 13034, 2948, 71024, 68023, 279, 22866, 3731, 1486, 62610, 2293, 11322, 4351, 1990, 32443, 16031, 28857, 8075, 56630, 523, 22609, 8, 2270, 72112, 78036, 537, 19903, 29283, 79300, 58299, 19339, 15089, 16703, 18793, 47205, 29508, 2367, 27818, 3895, 15384, 2942, 11987, 11078, 273, 3922, 31596, 4682, 10988, 1654, 62004, 6074, 1201, 6294, 524, 50669, 840, 15771, 2777, 790, 836, 3416, 55843, 183, 30177, 21187, 2793, 8854, 4796, 9925, 26, 14011, 448, 39232, 2464, 6779, 22564, 4398, 14689, 43154, 897, 66792, 16073, 13699, 403, 56116, 14026, 2369, 58306, 417, 56411, 66035, 1768, 13107, 25858, 1066, 569, 15873, 396, 41931, 13324, 24036, 8367, 7787, 1665, 3253, 18317, 15582, 3141, 12622, 3239, 10122, 3705, 51274, 9230, 39445, 52410, 2678, 58384, 25432, 2134, 72700, 1177, 12032, 8715, 29683, 29685, 16441, 185, 2007, 14371, 10854, 44669, 3183, 5104, 20251, 759]                                                                                                                                           |
# |25             |[5125, 6096, 19659, 5985, 1520, 7682, 8357, 191, 3061, 2369, 7163, 329, 34811, 207, 2312, 8811, 18001, 170, 642, 572, 1593, 24, 50, 1326, 1356, 1052, 44, 1864, 471, 1981, 211, 229, 7543, 5088, 692, 2490, 8746, 2559, 611, 25313, 14461, 52071, 1477, 3841, 481, 18844, 42185, 426, 14165, 2681, 1884, 5063, 520, 741, 13898, 1443, 698, 1691, 990, 20054, 113, 5580, 11450, 1535, 1480, 3696, 7149, 445, 1437, 733, 2655, 2128, 644, 1224, 381, 510, 9585, 4867, 2384, 3542, 6469, 8870, 12163, 4538, 2573, 479, 477, 13908, 858, 1289, 596, 812, 50256, 64709, 519, 239, 1233, 1769, 2944, 27927, 9470, 2519, 1137, 3967, 925, 1837, 2928, 22392, 2017, 8307, 13135, 214, 727, 10367, 557, 1320, 5510, 4080, 1227, 2350, 2095, 21510, 3016, 2104, 1702, 11104, 2060, 365, 2427, 37862, 64478, 243, 936, 3348, 6923, 34263, 12614, 31768, 3913, 7724, 6116, 55788, 12590, 506, 2330, 3202, 80042, 700, 22871, 21919, 31583, 3252, 3296, 22889, 49977, 2325, 22154, 25883, 757, 4289, 27174, 2064, 2754, 6996, 37527, 24455, 22029, 1405, 15426, 17991, 68077, 849, 49320, 3672, 4689, 587, 40154, 48818, 241, 23438, 893, 764, 22895, 6452, 44293, 9736, 51603, 1398, 22512, 13991, 4292, 3518, 8060, 6345, 1309, 10164, 10384, 1862, 6838, 1087, 636, 4314, 1184, 57304, 27762, 43499, 4603, 1631, 1980, 6298, 14265, 2374, 12518, 1646, 2446, 543, 1354, 10215, 6293, 171, 2971, 10495, 901, 2722, 907, 2466, 179, 1077, 348, 7917, 40550, 6412, 32480, 681, 951, 308, 3053, 756, 8613, 5592, 696, 120, 14176, 3600, 74885, 4153, 4877, 53728, 3736, 8614, 331, 40292, 5429, 4555, 6770, 3687, 7406, 1604, 3044, 6362, 8391, 6176, 2252, 1250, 28716, 4542, 20863, 215, 25284, 3097, 796, 47734, 1247, 1230, 7748, 65454, 3720, 21452, 563, 1723, 3218, 570, 9639, 2311, 1749, 4673, 71936, 48113, 161, 2150, 9730, 21205, 34, 32406, 713, 464, 3159, 1698, 11491]                    |
# |38             |[236, 8320, 4270, 50142, 32416, 20340, 30716, 55730, 273, 62705, 15550, 30737, 71654, 5986, 14393, 17821, 3377, 8699, 325, 4155, 20951, 16923, 13130, 416, 8521, 6079, 5824, 11189, 1584, 10319, 4986, 18580, 6881, 2732, 36357, 1521, 17384, 4524, 14694, 18145, 218, 14592, 11949, 4247, 23534, 6516, 4668, 2867, 3320, 9501, 606, 7750, 2395, 19145, 4180, 38339, 16514, 17123, 33668, 350, 13363, 33347, 36288, 20593, 9386, 14187, 13657, 5292, 11879, 14048, 28739, 7720, 45347, 5778, 22478, 3391, 31732, 2688, 443, 60704, 8817, 15623, 43499, 32299, 1357, 36592, 77392, 40261, 10233, 1878, 5189, 64296, 7494, 52001, 31774, 38426, 11222, 34553, 29020, 27866, 11927, 20698, 17632, 33863, 45797, 9324, 27770, 3153, 13309, 38705, 52384, 8187, 23797, 6925, 37607, 20407, 38323, 4736, 9543, 17260, 5445, 9394, 11111, 4005, 29522, 2981, 16331, 5930, 4855, 27243, 24561, 190, 6375, 11208, 30685, 29229, 11329, 38143, 20925, 29215, 33389, 30074, 11751, 21176, 48653, 3782, 4480, 13293, 11180, 12683, 45500, 44672, 3375, 25452, 22510, 996, 40788, 19684, 13999, 3810, 17081, 19052, 9976, 27994, 6944, 33712, 9705, 8911, 26679, 25489, 39637, 14761, 3426, 41668, 2731, 20084, 42124, 4947, 2294, 36526, 11603, 46801, 4108, 29980, 11146, 19113, 16472, 16515, 30314, 4056, 39531, 12765, 32855, 9297, 20355, 34482, 17160, 52705, 11322, 41790, 28207, 40891, 13590, 12574, 16217, 67671, 1790, 12784, 20787, 12862, 27535, 20387, 12903, 10910, 14174, 72753, 19288, 2284, 48665, 35251, 4738, 19109, 19279, 37596, 128, 23007, 2865, 20666, 61967, 16761, 29645, 19890, 10764, 33409, 7868, 19633]                                                                                                                                                                                                                                                                 |
# |46             |[16109, 41486, 8205, 5119, 15442, 955, 16889, 1828, 22429, 3216, 8291, 1224, 15890, 241, 179, 44082, 21213, 3230, 50866, 6153, 27001, 24262, 48423, 53404, 44066, 4041, 31572, 35007, 50652, 65353, 9774, 34105, 7874, 14767, 16334, 15902, 2456, 3396, 36137, 22223, 189, 2157, 74036, 12502, 77004, 1791, 69694, 54080, 910, 36520, 1810, 13461, 13026, 29351, 77360, 53161, 52983, 23931, 6263, 33550, 14296, 14531, 68642, 28667, 18294, 56579, 489, 19696, 1262, 1489, 19874, 440, 4937, 50062, 1377, 66354, 4513, 4867, 29646, 13380, 1836, 17112, 45873, 29449, 12940, 280, 6458, 8246, 7246, 55803, 59, 52448, 800, 20967, 13351, 2429, 7623, 8348, 5129, 419, 15384, 23166, 403, 21903, 21890, 42433, 63576, 2985, 3573, 24317, 19366, 24254, 36674, 1022, 23772, 273, 15124, 3052, 6093, 26978, 811, 2137, 2972, 29946, 63338, 62557, 29145, 2155, 769, 6092, 3544, 12273, 55040, 25735, 47454, 6375, 9847, 2106, 26998, 5344, 21298, 18433, 13400, 49698, 16148, 18989, 2756, 64901, 68639, 1324, 13652, 20119, 11811, 40172, 6289, 7890, 204, 9223, 8650, 50230, 52115, 802, 45518, 70629, 30845, 1727, 29938, 81196, 76187, 13297, 2274, 18532, 8845, 46245, 2180, 1499, 7097, 7006, 17865, 1787, 1293, 248, 14698, 12653, 67010, 57079, 53935, 1252, 63518, 700, 1585, 363, 1627, 54995, 6014, 649, 9231, 2076, 27666, 1961, 4440, 588, 652, 8935, 7089, 53694, 56851, 76005, 3064, 19180, 11876, 20255, 14723]                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
# |50             |[1026, 1265, 5035, 1872, 4891, 25224, 9509, 8115, 9355, 22427, 19515, 15214, 18621, 21091, 358, 5951, 12405, 455, 15409, 248, 279, 23741, 15359, 15142, 3017, 17010, 3290, 7601, 4906, 5966, 10969, 32879, 22095, 15842, 8957, 21973, 14211, 4403, 41433, 11282, 16197, 10398, 10439, 9655, 20396, 11231, 857, 2487, 15839, 22998, 8837, 11280, 36015, 9677, 3963, 6897, 29569, 3953, 30676, 14512, 7592, 4029, 15213, 14812, 11177, 4607, 4453, 27122, 15791, 19284, 14701, 555, 7692, 17787, 7785, 37703, 8752, 22723, 26257, 14483, 29912, 16848, 21039, 671, 18574, 4015, 29886, 1526, 6058, 42256, 2767, 1818, 15135, 29340, 21258, 843, 14948, 23496, 30189, 7891, 13610, 13527, 16184, 20811, 10847, 3666, 787, 30885, 7607, 37061, 23552, 249, 31002, 3343, 7501, 64110, 5883, 620, 13751, 29159, 6937, 27569, 8038, 415, 48831, 1583, 31233, 14185, 514, 43881, 10476, 14847, 3336, 17641, 30516, 5076, 10221, 1674, 8854, 16356, 3352, 8778, 5639, 9514, 2936, 1926, 10132, 22629, 9904, 15849, 6031, 13054, 573, 17528, 29615, 52209, 56484, 590, 12376, 19391, 3395, 41, 16812, 40217, 8769, 9689, 276, 13320, 400, 1982, 2749, 10567, 4519, 10510, 5847, 69, 43284, 64237, 8035, 20799, 10834, 3818, 11343, 39205, 2998, 51773, 5477, 12893, 9730, 44552, 15073, 6136, 15105, 27321, 12246, 10512, 7373, 7969, 3146, 6398, 26764, 18262, 28621, 13022, 1190, 4827, 4389, 4522, 9678, 19810, 27166, 31602, 4287, 25579, 79495, 21872, 50206, 4418, 27836, 1919, 59181, 3799, 381]                                                                                                                                                                                                                                                                                                                                                                                              |
# |73             |[7737, 8135, 13286, 8538, 24198, 21868, 544, 8079, 9099, 43178, 17837, 2981, 1267, 11575, 7310, 629, 32600, 8823, 44350, 26061, 821, 262, 17590, 3970, 146, 627, 3034, 4942, 1713, 9910, 37111, 48417, 453, 9428, 4965, 18189, 1164, 14113, 15490, 80846, 5277, 3128, 38449, 319, 35825, 3260, 2242, 9450, 24544, 10246, 1260, 5479, 41575, 33188, 33918, 16750, 24272, 1998, 18816, 5869, 26358, 9066, 56268, 16032, 12365, 15221, 2417, 1125, 2604, 40135, 19871, 13529, 40425, 18388, 24804, 5691, 24633, 10372, 16618, 48648, 12430, 9170, 16230, 40261, 2999, 5508, 4279, 31359, 18065, 37110, 9689, 28907, 4116, 7157, 1705, 3189, 11888, 8783, 20943, 52789, 15214, 54284, 54510, 33146, 56802, 33959, 8992, 18428, 3205, 8877, 37735, 695, 57219, 27064, 4041, 11524, 12796, 7061, 13828, 680, 5778, 10226, 75458, 4985, 180, 1457, 9307, 443, 1931, 12856, 41573, 6042, 2404, 2990, 5157, 32414, 3394, 19391, 13620, 7043, 26209, 35394, 15663, 1633, 33710, 23627, 18996, 7716, 8886, 1750, 55550, 18432, 24254, 35791, 2316, 24446, 8296, 26200, 14285, 32617, 25140, 18747, 14738, 878, 13013, 2641, 39227, 50529, 5402, 2113, 7407, 1050, 16022, 3314, 641, 41205, 2828, 2599, 28058, 25681, 432, 2008, 32080, 13033, 33102, 1308, 6919, 51989, 69425, 25437, 1258, 23010, 27938, 10121, 35994, 3163, 7185, 11905, 12756, 1355, 37666, 4843, 24138]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
# |97             |[10970, 27174, 18223, 13417, 17120, 13327, 52123, 38963, 32585, 7104, 22175, 37319, 26506, 19652, 12517, 16850, 38703, 29807, 10610, 36950, 63492, 65221, 8449, 3100, 10598, 33119, 70486, 11058, 52243, 54877, 8841, 33288, 38823, 40630, 574, 9880, 25555, 3948, 1458, 356, 233, 36032, 44000, 60997, 41604, 8055, 6887, 10643, 31016, 3271, 48972, 14681, 7541, 23844, 4026, 3040, 1877, 351, 25478, 5421, 762, 7339, 2907, 150, 25381, 56283, 7675, 13378, 4713, 2972, 21701, 30385, 9922, 5397, 270, 4094, 25714, 49347, 531, 17891, 4489, 59697, 2028, 51624, 2218, 1622, 30043, 55755, 18926, 3534, 1474, 5121, 280, 12907, 55715, 88, 8005, 43263, 45100, 13, 1807, 4053, 859, 19501, 24466, 1436, 30932, 1828, 17666, 13033, 579, 7884, 30989, 4027, 24856, 4394, 60905, 16639, 11234, 48616, 10128, 5533, 32698, 56980, 66385, 46259, 628, 20215, 2809, 12002, 6170, 9663, 7074, 25745, 5592, 2184, 2119, 23110, 21182, 20111, 8082, 77944, 28853, 24170, 14440, 42618, 6075, 13419, 3603, 3598, 6107, 6446, 1556, 21968, 60175, 3866, 28396, 90, 1100, 67278, 60144, 4145, 4102, 40872, 124, 2354, 13843, 17280, 27488, 47753, 77498, 898, 5149, 45760, 6009, 20874, 980, 6775, 10138, 23668, 27667, 41602, 18045, 10294, 14091, 1256, 40878, 374, 17773, 29757, 45340, 2239, 5473, 79328, 218, 3354, 76372, 5894, 3552, 45980, 6814, 66788, 3618, 69797, 393, 220, 18393, 77122, 72606, 64001, 28157, 311, 40867, 18747]                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
# |161            |[8077, 5221, 684, 4230, 5499, 15326, 385, 7807, 14976, 4981, 79909, 3754, 11228, 5269, 4709, 9248, 48058, 49319, 18409, 9855, 392, 5043, 3741, 14595, 251, 2965, 10725, 17939, 2499, 16056, 5995, 4093, 2397, 11874, 191, 6818, 18894, 7452, 422, 1388, 3970, 10082, 4590, 31964, 200, 8907, 5167, 36477, 29635, 6194, 681, 3471, 338, 8019, 2226, 2402, 6823, 426, 64805, 3849, 1548, 28897, 7241, 3615, 10815, 5902, 15045, 21246, 10668, 3702, 5944, 10493, 4853, 53050, 1326, 563, 63082, 15086, 61099, 4690, 12713, 8771, 36155, 9769, 25404, 12102, 21409, 14056, 14613, 6401, 11810, 51305, 595, 8712, 41402, 12548, 1255, 1936, 16050, 16199, 8162, 10107, 1845, 47543, 1507, 16427, 434, 8340, 17806, 3712, 37369, 6605, 567, 3890, 20830, 70498, 11028, 14511, 543, 15220, 16480, 16428, 16680, 4890, 21927, 37638, 14220, 38494, 5757, 89, 17121, 5229, 28327, 740, 8262, 22862, 10308, 10684, 14036, 14992, 6387, 7888, 34128, 1475, 46135, 41722, 21484, 2602, 7921, 26657, 8910, 28577, 45068, 1544, 8096, 2350, 65887, 11500, 78926, 2673, 1777, 6873, 43615, 5982, 5136, 53942, 34650, 5802, 5879, 25593, 6970, 508, 2191, 2802, 15223, 13644]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
# |172            |[5158, 9517, 13472, 14750, 17000, 11096, 9899, 20977, 21335, 3739, 14419, 26315, 14561, 10448, 35672, 11462, 19712, 7105, 20082, 18953, 24721, 49892, 3436, 18066, 709, 7399, 7580, 8028, 13316, 33567, 45890, 11415, 35054, 23008, 15340, 44544, 13372, 13055, 14368, 19228, 10126, 4594, 1463, 9986, 27025, 2214, 7359, 7598, 19164, 10694, 71102, 13586, 10351, 14230, 11586, 11891, 536, 6136, 15350, 3748, 45358, 7884, 31442, 8684, 43306, 36696, 23713, 33562, 1665, 3361, 23564, 37676, 934, 8786, 35011, 20888, 1750, 5778, 42747, 43035, 30422, 28860, 7179, 1050, 32998, 7808, 28548, 32403, 2628, 14523, 30955, 28553, 1321, 31834, 24922, 50007, 23161, 1067, 31321, 9913, 66070, 11414, 24105, 2406, 67410, 24282, 3167, 1499, 18257, 1095, 14234, 39861, 36704, 6669, 7918, 7198, 25941, 8356, 33726, 41350, 3385, 48787, 11906, 40321, 1477, 4541, 13424, 58293, 64666, 10915, 39045, 76490, 716, 18173, 66391, 2597, 53516, 75259, 29846, 22778, 383, 4766, 36782, 52605, 32441, 845, 13174, 41886, 2530, 13366, 22457, 35719, 55282, 6694, 9991, 31808, 1657, 1378, 26092, 516, 18465, 14143, 5990, 878, 16290, 1157, 265, 22829, 6777, 47773, 31618, 21428, 19038, 6888, 5653, 19478, 14634, 66950, 22548, 36205, 6166, 19012, 42963, 52, 30163, 22764, 36387, 19684, 21178, 203, 8756]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
# +---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# only showing top 10 rows


combined = (
    recommended_songs.join(relevant_songs, on='user_id_encoded', how='inner')
    .rdd
    .map(lambda row: (row[1], row[2]))
)
combined.cache()
combined.count()

# 298373

combined.take(1)



# [([2, 12, 0, 9, 8],
  # [8958,
   # 41647,
   # 1253,
   # 48312,
   # 51149,
   # 39234,
   # 11321,
   # 88,
   # 11003,
   # 3701,
   # 14207,
   # 62906,
   # 11464,
   # 36159,
   # 73110,
   # 49657,
   # 2443,
   # 25586,
   # 4329,
   # 7,
   # 4471,
   # 15096,
   # 22512,
   # 35070,
   # 9259,
   # 13533,
   # 13685,
   # 17900,
   # 11268,
   # 27709,
   # 4877,
   # 8619,
   # 64,
   # 23827,
   # 11477,
   # 4496,
   # 1008,
   # 12339,
   # 5582,
   # 11250,
   # 9599,
   # 5122,
   # 10362,
   # 12092,
   # 17449,
   # 25119,
   # 15981,
   # 14487,
   # 16571,
   # 11545,
   # 1565,
   # 13837,
   # 63288,
   # 61325,
   # 6604,
   # 7389,
   # 62342,
   # 17633,
   # 48992,
   # 47253,
   # 38871,
   # 5994,
   # 498,
   # 59306,
   # 8840,
   # 17485,
   # 19064,
   # 58225,
   # 261,
   # 62565,
   # 2744,
   # 657,
   # 58363,
   # 20580,
   # 2840,
   # 689,
   # 43569,
   # 1927,
   # 5800,
   # 45785,
   # 20995,
   # 39360,
   # 24194,
   # 6265,
   # 5097,
   # 5736,
   # 43492,
   # 69589,
   # 28868,
   # 11957,
   # 1505,
   # 42046,
   # 25315,
   # 274,
   # 15411,
   # 110,
   # 41797,
   # 435,
   # 942,
   # 24295,
   # 11406,
   # 15061,
   # 14790,
   # 26816,
   # 12289,
   # 27913,
   # 7654,
   # 41369,
   # 1048,
   # 15222,
   # 10165,
   # 2988,
   # 5772,
   # 24078,
   # 36070,
   # 4794,
   # 2065,
   # 35503,
   # 3422,
   # 1520,
   # 65077,
   # 26827,
   # 27482,
   # 1019,
   # 16665,
   # 9987,
   # 459,
   # 2238,
   # 241,
   # 6000,
   # 781,
   # 163,
   # 9380,
   # 2960,
   # 55846,
   # 15510,
   # 7282,
   # 2515,
   # 23345,
   # 15498,
   # 1309,
   # 7351,
   # 5487,
   # 9462,
   # 1715,
   # 24142,
   # 11686,
   # 8490,
   # 14066,
   # 53729,
   # 4762,
   # 226,
   # 1819,
   # 4490,
   # 5612,
   # 9652,
   # 9210,
   # 12437,
   # 14571,
   # 3903,
   # 3425,
   # 12783,
   # 34226,
   # 22509,
   # 23577,
   # 19569,
   # 7831,
   # 5315,
   # 2114,
   # 6795,
   # 6758,
   # 33376,
   # 3849,
   # 6744,
   # 2282,
   # 5872,
   # 15145,
   # 423,
   # 37195,
   # 26595,
   # 16698,
   # 237,
   # 5544,
   # 3895,
   # 22906,
   # 27097,
   # 1013,
   # 35572,
   # 4479,
   # 15005,
   # 67,
   # 3393,
   # 19726,
   # 6193,
   # 14084,
   # 2255,
   # 2456,
   # 7338,
   # 29443,
   # 19362,
   # 18481,
   # 13102,
   # 6129,
   # 60662,
   # 7819,
   # 7742,
   # 36987,
   # 15106,
   # 16712,
   # 538,
   # 3113,
   # 6956,
   # 3769,
   # 66601,
   # 77993,
   # 27113,
   # 26654,
   # 346,
   # 15001,
   # 3343,
   # 47947,
   # 47856,
   # 132,
   # 2937,
   # 3118,
   # 16385,
   # 24733,
   # 8607,
   # 12831,
   # 1403,
   # 9344,
   # 18759,
   # 58966,
   # 7176,
   # 31269,
   # 6047,
   # 1196,
   # 23280,
   # 164,
   # 920,
   # 1023,
   # 9826,
   # 3850,
   # 12366,
   # 28789,
   # 39751,
   # 238,
   # 29634,
   # 7792,
   # 81294,
   # 2681,
   # 14634,
   # 29643,
   # 30253,
   # 42982,
   # 27163,
   # 3836,
   # 8907,
   # 10371,
   # 43449,
   # 11279,
   # 7123,
   # 4368,
   # 4426,
   # 76208,
   # 16626,
   # 28336,
   # 29635,
   # 4407,
   # 53979,
   # 18437,
   # 38800,
   # 518,
   # 11117,
   # 12007,
   # 31875,
   # 19200,
   # 1659,
   # 40959,
   # 864,
   # 11371,
   # 1838,
   # 13138])]



rankingMetrics = RankingMetrics(combined)
ndcgAtK = rankingMetrics.ndcgAt(k)
print(ndcgAtK) # 0.033265494148292274



