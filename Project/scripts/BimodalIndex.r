library(BimodalIndex)

# import dataset
# - rows represent the different proteins
# - columns represent the realizations
data <- read.csv("../data/data_LN_PST_MB.csv", header = FALSE, fill=TRUE, strip.white=TRUE)
prot_names <- data[2:dim(data)[1],1]
data <- data.frame(data[2:dim(data)[1],2:dim(data)[2]])
rownames(data) <- prot_names
colnames(data) <- seq(1:dim(data)[2])


# select datasets before and after the differentiation
nondiff_data = data[,1:floor(dim(data)[2]*(1.0/6))]
diff_data = data[,floor(dim(data)[2]*(5.0/6)):dim(data)[2]]


# compute bimodal index with the dedicated library
bi_nondiff_df <- bimodalIndex(nondiff_data, verbose=FALSE)
bi_diff_df <- bimodalIndex(diff_data, verbose=FALSE)
colnames(bi_nondiff_df) <- paste(colnames(bi_nondiff_df),"nondiff",sep="_")
colnames(bi_diff_df) <- paste(colnames(bi_diff_df),"diff",sep="_")


# export computed bimodal index
bi_df <- cbind(bi_nondiff_df, bi_diff_df)
write.csv(bi_df, "../results/BimodalIndex_result.csv")