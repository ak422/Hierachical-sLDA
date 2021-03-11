// (C) Copyright 2009, Chong Wang, David Blei and Li Fei-Fei

// written by Chong Wang, chongw@cs.princeton.edu

// This file is part of slda.

// slda is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// slda is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

#include "slda.h"
#include <time.h>
#include "utils.h"
#include "assert.h"
#include "opt.h"
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

const int NUM_INIT = 50;
const int LAG = 10;
const int LDA_INIT_MAX = 0;
const int MSTEP_MAX_ITER = 50;

slda::slda()
{
    //ctor
    alpha = 1.0;
    num_topics = 0;
    num_classes = 0;
    size_vocab = 0;
    size_feature = 0;

    log_prob_w = NULL;
    log_prob_f = NULL;
    eta = NULL;
}

slda::~slda()
{
    free_model();
}

/*
 * init the model
 */

void slda::init(double alpha_, int num_topics_,
                const corpus * c)
{
    alpha = alpha_;                 //超参数
    num_topics = num_topics_;       //主题个数
    size_vocab = c->size_vocab;     //特征值列表长度
    size_feature = c->size_feature;     //特征列表长度
    num_classes = c->num_classes;  //分类数目

    log_prob_w = new double * [num_topics]; //the log of the topic distribution
    for (int k = 0; k < num_topics; k++)
    {
        log_prob_w[k] = new double [size_vocab];  //每一个主题分布都是在整个词汇列表上
        memset(log_prob_w[k], 0, sizeof(double)*size_vocab);  //初始化为0
    }

    // by ak422
    log_prob_f = new double * [num_topics]; //the log of the topic distribution
    for (int k = 0; k < num_topics; k++)
    {
        log_prob_f[k] = new double [size_feature];  //每一个主题分布都是在整个词汇列表上
        memset(log_prob_f[k], 0, sizeof(double)*size_feature);  //初始化为0
    }


    //no need to train slda if we only have one class
    if (num_classes > 1)
    {
        eta = new double * [num_classes-1];  //num_classes-1 个分类的eta参数
        for (int i = 0; i < num_classes-1; i ++)
        {
            eta[i] = new double [num_topics];  //每个分类都在所有主题上有一个系数
            memset(eta[i], 0, sizeof(double)*num_topics);
        }
    }
}

/*
 * free the model
 */

void slda::free_model()
{
    if (log_prob_w != NULL)
    {
        for (int k = 0; k < num_topics; k++)
        {
            delete [] log_prob_w[k];
        }
        delete [] log_prob_w;
        log_prob_w = NULL;
    }

    if (log_prob_f != NULL)
    {
        for (int k = 0; k < num_topics; k++)
        {
            delete [] log_prob_f[k];
        }
        delete [] log_prob_f;
        log_prob_f = NULL;
    }




    if (eta != NULL)
    {
        for (int i = 0; i < num_classes-1; i ++)
        {
            delete [] eta[i];
        }
        delete [] eta;
        eta = NULL;
    }
}

/*
 * save the model in the binary format
 */

void slda::save_model(const char * filename)
{
    FILE * file = NULL;
    file = fopen(filename, "wb");
    fwrite(&alpha, sizeof (double), 1, file);
    fwrite(&num_topics, sizeof (int), 1, file);
    fwrite(&size_vocab, sizeof (int), 1, file);
    fwrite(&size_feature, sizeof (int), 1, file);
    fwrite(&num_classes, sizeof (int), 1, file);

    for (int k = 0; k < num_topics; k++)
    {
        fwrite(log_prob_w[k], sizeof(double), size_vocab, file);
    }

    for (int k = 0; k < num_topics; k++)
    {
        fwrite(log_prob_f[k], sizeof(double), size_feature, file);
    }



    if (num_classes > 1)
    {
        for (int i = 0; i < num_classes-1; i ++)
        {
            fwrite(eta[i], sizeof(double), num_topics, file);
        }
    }

    fflush(file);
    fclose(file);
}

/*
 * load the model in the binary format
 */

void slda::load_model(const char * filename)
{
    FILE * file = NULL;
    file = fopen(filename, "rb");
    fread(&alpha, sizeof (double), 1, file);
    fread(&num_topics, sizeof (int), 1, file);
    fread(&size_vocab, sizeof (int), 1, file);
    fread(&size_feature, sizeof (int), 1, file);
    fread(&num_classes, sizeof (int), 1, file);

    //读取模型中 主题-单词分布的log
    log_prob_w = new double * [num_topics];
    for (int k = 0; k < num_topics; k++)
    {
        log_prob_w[k] = new double [size_vocab];
        fread(log_prob_w[k], sizeof(double), size_vocab, file);
    }


    //读取模型中 主题-特征分布的log
    log_prob_f = new double * [num_topics];
    for (int k = 0; k < num_topics; k++)
    {
        log_prob_f[k] = new double [size_feature];
        fread(log_prob_f[k], sizeof(double), size_feature, file);
    }


    //读取模型中 主题的softmax回归系数
    if (num_classes > 1)
    {
        eta = new double * [num_classes-1];
        for (int i = 0; i < num_classes-1; i ++)
        {
            eta[i] = new double [num_topics];
            fread(eta[i], sizeof(double), num_topics, file);
        }
    }

    fflush(file);
    fclose(file);
}

/*
 * save the model in the text format
 */

void slda::save_model_text(const char * filename)
{
    FILE * file = NULL;
    file = fopen(filename, "w");
    fprintf(file, "alpha: %lf\n", alpha);
    fprintf(file, "number of topics: %d\n", num_topics);
    fprintf(file, "size of vocab: %d\n", size_vocab);
    fprintf(file, "size of feature: %d\n", size_feature);
    fprintf(file, "number of classes: %d\n", num_classes);

    fprintf(file, "betas: \n"); // in log space
    for (int k = 0; k < num_topics; k++)
    {
        for (int j = 0; j < size_vocab; j ++)
        {
            fprintf(file, "%lf ", log_prob_w[k][j]);
        }
        fprintf(file, "\n");
    }

    fprintf(file, "betas_2: \n"); // in log space
    for (int k = 0; k < num_topics; k++)
    {
        for (int j = 0; j < size_feature; j ++)
        {
            fprintf(file, "%lf ", log_prob_f[k][j]);
        }
        fprintf(file, "\n");
    }


    if (num_classes > 1)
    {
        fprintf(file, "etas: \n");
        for (int i = 0; i < num_classes-1; i ++)
        {
            for (int j = 0; j < num_topics; j ++)
            {
                fprintf(file, "%lf ", eta[i][j]);
            }
            fprintf(file, "\n");
        }
    }

    fflush(file);
    fclose(file);
}

/*
 * create the data structure for sufficient statistic 
 */
//充分统计量
suffstats * slda::new_suffstats(int num_docs)
{
    suffstats * ss = new suffstats;
    ss->num_docs = num_docs;        //文档个数

    ss->word_total_ss = new double [num_topics];  //每个主题的特征值统计
    memset(ss->word_total_ss, 0, sizeof(double)*num_topics);  //每个主题的特征值统计初始化为0
    
    ss->word_ss = new double * [num_topics];  //每个主题下的词汇表的分布
    for (int k = 0; k < num_topics; k ++)
    {
        ss->word_ss[k] = new double [size_vocab];
        memset(ss->word_ss[k], 0, sizeof(double)*size_vocab);  //每个主题-特征值分布初始化为0 
    }


    ss->word_total_ff = new double [num_topics];  //每个主题的特征统计
    memset(ss->word_total_ff, 0, sizeof(double)*num_topics);  //每个主题的特征统计初始化为0
   
    ss->word_ff = new double * [num_topics];  //每个主题下的词汇表的分布
    for (int k = 0; k < num_topics; k ++)
    {
        ss->word_ff[k] = new double [size_feature];
        memset(ss->word_ff[k], 0, sizeof(double)*size_feature);  //每个主题分布-特征初始化为0 
    }



    int num_var_entries = num_topics*(num_topics+1)/2;
    ss->z_bar =  new z_stat [num_docs];
    for (int d = 0; d < num_docs; d ++)
    {
        ss->z_bar[d].z_bar_m = new double [num_topics];  //每个文档下的主题分布
        ss->z_bar[d].z_bar_var = new double [num_var_entries];  //每个文档有num_topics*(num_topics+1)/2个
        memset(ss->z_bar[d].z_bar_m, 0, sizeof(double)*num_topics);
        memset(ss->z_bar[d].z_bar_var, 0, sizeof(double)*num_var_entries);
    }
    ss->labels = new int [num_docs];   //每个文档一个标签
    memset(ss->labels, 0, sizeof(int)*(num_docs));
    ss->tot_labels = new int [num_classes];  //按标签计数统计
    memset(ss->tot_labels, 0, sizeof(int)*(num_classes));

    return(ss);
}


/*
 * initialize the sufficient statistics with zeros
 */

void slda::zero_initialize_ss(suffstats * ss)
{
    memset(ss->word_total_ss, 0, sizeof(double)*num_topics);
    for (int k = 0; k < num_topics; k ++)
    {
        memset(ss->word_ss[k], 0, sizeof(double)*size_vocab);
    }


    memset(ss->word_total_ff, 0, sizeof(double)*num_topics);
    for (int k = 0; k < num_topics; k ++)
    {
        memset(ss->word_ff[k], 0, sizeof(double)*size_feature);
    }



    int num_var_entries = num_topics*(num_topics+1)/2;
    for (int d = 0; d < ss->num_docs; d ++)
    {
        memset(ss->z_bar[d].z_bar_m, 0, sizeof(double)*num_topics);
        memset(ss->z_bar[d].z_bar_var, 0, sizeof(double)*num_var_entries);
    }
    ss->num_docs = 0;
}


/*
 * initialize the sufficient statistics with random numbers 
 */



// 随机初始化主题-词汇表分布，和文档-主题分布
void slda::random_initialize_ss(suffstats * ss, corpus* c)  
{
    int num_docs = ss->num_docs;
    gsl_rng * rng = gsl_rng_alloc(gsl_rng_taus);
    time_t seed;
    time(&seed);
    gsl_rng_set(rng, (long) seed);
    int k, w, d, j, idx;
    for (k = 0; k < num_topics; k++)
    {
        for (w = 0; w < size_vocab; w++)
        {
            ss->word_ss[k][w] = 1.0/size_vocab + 0.1*gsl_rng_uniform(rng);
            ss->word_total_ss[k] += ss->word_ss[k][w];  //word_ss[k][w]：主题-词汇表分布
        }

        for (w = 0; w < size_feature; w++)
        {
            ss->word_ff[k][w] = 1.0/size_feature + 0.1*gsl_rng_uniform(rng);
            ss->word_total_ff[k] += ss->word_ff[k][w];  //word_ss[k][w]：主题-词汇表分布
        }   
    }

    for (d = 0; d < num_docs; d ++)
    {
        document * doc = c->docs[d];
        ss->labels[d] = doc->label;
        ss->tot_labels[doc->label] ++;

        double total = 0.0;
        for (k = 0; k < num_topics; k ++)
        {
            ss->z_bar[d].z_bar_m[k] = gsl_rng_uniform(rng);
            total += ss->z_bar[d].z_bar_m[k];
        }
        for (k = 0; k < num_topics; k ++)
        {
            ss->z_bar[d].z_bar_m[k] /= total;
        }
        for (k = 0; k < num_topics; k ++)
        {
            for (j = k; j < num_topics; j ++)
            {
                idx = map_idx(k, j, num_topics);
                if (j == k)
                    ss->z_bar[d].z_bar_var[idx] = ss->z_bar[d].z_bar_m[k] / (double)(doc->total);
                else
                    ss->z_bar[d].z_bar_var[idx] = 0.0;

                ss->z_bar[d].z_bar_var[idx] -=
                    ss->z_bar[d].z_bar_m[k] * ss->z_bar[d].z_bar_m[j] / (double)(doc->total);
            }
        }
    }

    gsl_rng_free(rng);
}


// 根据语料库，即真实数据初始化主题-词汇表分布，和文档-主题分布（未对所有文档扫描，是一个估计值）
void slda::corpus_initialize_ss(suffstats* ss, corpus* c)
{
    int num_docs = ss->num_docs;
    gsl_rng * rng = gsl_rng_alloc(gsl_rng_taus);
    time_t seed;
    time(&seed);
    gsl_rng_set(rng, (long) seed);
    int k, n,m, d, j, idx, i, w;

    for (k = 0; k < num_topics; k++)
    {
        for (i = 0; i < NUM_INIT; i++)  //随机选择某个文档操作的次数，即NUM_INIT次随机抽取（每个主题NUM_INIT=50次随机选择文档计算）
        {
            d = (int)(floor(gsl_rng_uniform(rng) * num_docs)); //产生一个在[0, 1)区间上的双精度随机数，它产生的随机数包括0，但不包括1。
            printf("initialized with document %d\n", d);//随机选择一个文档

            document * doc = c->docs[d];
            for (n = 0; n < doc->length; n++)
            {
                //doc->words[n]：文档第n个词是词汇列表的第几个词
                //ss->word_ss[k][doc->words[n]]：第k个主题，第doc->words[n]词的总数
                ss->word_ss[k][doc->words[n]] += doc->counts[n]; //统计文档中每个主题下-按词汇列表统计求和

            }


            document * docs_feature = c->docs_feature[d];
            for (m = 0; m < docs_feature->length; m++)
            {
                //doc->words[n]：文档第n个词是词汇列表的第几个词
                //ss->word_ss[k][doc->words[n]]：第k个主题，第doc->words[n]词的总数
                ss->word_ff[k][docs_feature->words[m]] += docs_feature->counts[m]; //统计文档中每个主题下-按词汇列表统计求和
            }            

        }

        for (w = 0; w < size_vocab; w++) //每个主题下，词汇列表中的每个词
        {
            ss->word_ss[k][w] = 2*ss->word_ss[k][w] + 5 + gsl_rng_uniform(rng); // by removing words that occur in fewer than 5 documents.
            ss->word_total_ss[k] = ss->word_total_ss[k] + ss->word_ss[k][w]; //统计每个主题下的词个数
        }


        for (w = 0; w < size_feature; w++) //每个主题下，词汇列表中的每个词
        {
            ss->word_ff[k][w] = 2*ss->word_ff[k][w] + 5 + gsl_rng_uniform(rng); // by removing words that occur in fewer than 5 documents.
            ss->word_total_ff[k] = ss->word_total_ff[k] + ss->word_ff[k][w]; //统计每个主题下的词个数
        }
    }

    for (d = 0; d < num_docs; d ++)
    {
        document * doc = c->docs[d];
        ss->labels[d] = doc->label;
        ss->tot_labels[doc->label] ++;

        double total = 0.0;
        for (k = 0; k < num_topics; k ++)
        {
            ss->z_bar[d].z_bar_m[k] = gsl_rng_uniform(rng);
            total += ss->z_bar[d].z_bar_m[k];
        }
        for (k = 0; k < num_topics; k ++)
        {
            ss->z_bar[d].z_bar_m[k] /= total;
        }
        for (k = 0; k < num_topics; k ++)
        {
            for (j = k; j < num_topics; j ++)
            {
                idx = map_idx(k, j, num_topics);
                if (j == k)
                    ss->z_bar[d].z_bar_var[idx] = ss->z_bar[d].z_bar_m[k] / (double)(doc->total);
                else
                    ss->z_bar[d].z_bar_var[idx] = 0.0;

                ss->z_bar[d].z_bar_var[idx] -=
                    ss->z_bar[d].z_bar_m[k] * ss->z_bar[d].z_bar_m[j] / (double)(doc->total);
            }
        }
    }
    gsl_rng_free(rng);
}

void slda::load_model_initialize_ss(suffstats* ss, corpus * c)
{
    int num_docs = ss->num_docs;                                                                         
    for (int d = 0; d < num_docs; d ++)   //根据语料库初始化充分统计量的标签    
    {                                                                                                    
       document * doc = c->docs[d];     //每个文档一个标签
       ss->labels[d] = doc->label;
       ss->tot_labels[doc->label] ++;   //统计每个标签包括几个文档，即每个标签的计数
    }     
}

void slda::free_suffstats(suffstats * ss)
{
    delete [] ss->word_total_ss;

    for (int k = 0; k < num_topics; k ++)
    {
        delete [] ss->word_ss[k];
    }
    delete [] ss->word_ss;

    for (int d = 0; d < ss->num_docs; d ++)
    {
        delete [] ss->z_bar[d].z_bar_m;
        delete [] ss->z_bar[d].z_bar_var;
    }
    delete [] ss->z_bar;
    delete [] ss->labels;
    delete [] ss->tot_labels;

    delete ss;
}

void slda::v_em(corpus * c, const settings * setting,
                const char * start, const char * directory)
{
    char filename[100];
    int max_length = c->max_corpus_length(); //单个文档包括（指不同）词语个数最大值
    double **var_gamma, **phi, **lambda;
    double likelihood, likelihood_old = 0, converged = 1;
    int d, n, i;
    double L2penalty = setting->PENALTY;
    // allocate variational parameters
    var_gamma = new double * [c->num_docs];   //文档个数 ，每一个文档都有一个主题分布
    for (d = 0; d < c->num_docs; d++)
        var_gamma[d] = new double [num_topics];     //文档-主题分布：变分参数

    phi = new double * [max_length];  //每一个单词都有一个主题分布
    for (n = 0; n < max_length; n++)
        phi[n] = new double [num_topics];//：变分参数

    printf("initializing ...\n");
    suffstats * ss = new_suffstats(c->num_docs);//初始化充分统计量的数据结构
    if (strcmp(start, "seeded") == 0)  //seeded/random/model_path
    {
        corpus_initialize_ss(ss, c);
        mle(ss, 0, setting);//仅作参数初始化
    }
    else if (strcmp(start, "random") == 0)  //seeded/random/model_path
    {
        random_initialize_ss(ss, c);
        mle(ss, 0, setting);
    }
    else   //seeded/random/model_path
    {
        load_model(start);
        load_model_initialize_ss(ss, c);
    }

    FILE * likelihood_file = NULL;
    sprintf(filename, "%s/likelihood.dat", directory);
    likelihood_file = fopen(filename, "w");

    int ETA_UPDATE = 0;

    i = 0;
    while (((converged < 0) || (converged > setting->EM_CONVERGED) || (i <= LDA_INIT_MAX+2)) && (i <= setting->EM_MAX_ITER))
    {
        printf("**** em iteration %d ****\n", ++i);
        likelihood = 0;
        zero_initialize_ss(ss);
        if (i > LDA_INIT_MAX) ETA_UPDATE = 1;
        // e-step
        printf("**** e-step ****\n");
        for (d = 0; d < c->num_docs; d++)  //对每个文档执行E-STEP
        {
            if ((d % 100) == 0) printf("document %d\n", d);
            likelihood += doc_e_step(c->docs[d],c->docs_feature[d], var_gamma[d], phi, ss, ETA_UPDATE, setting); //ETA_UPDATE =1 :sLDA 推断，ETA_UPDATE =0 :LDA 推断
        }

        printf("likelihood: %10.10f\n", likelihood);
        // m-step
        printf("**** m-step ****\n");
        mle(ss, ETA_UPDATE, setting);

        // check for convergence
        converged = fabs((likelihood_old - likelihood) / (likelihood_old));
        //if (converged < 0) VAR_MAX_ITER = VAR_MAX_ITER * 2;
        likelihood_old = likelihood;

        // output model and likelihood
        fprintf(likelihood_file, "%10.10f\t%5.5e\n", likelihood, converged);
        fflush(likelihood_file);
        if ((i % LAG) == 0)
        {
            sprintf(filename, "%s/%03d.model", directory, i);
            save_model(filename);
            sprintf(filename, "%s/%03d.model.text", directory, i);
            save_model_text(filename);
            sprintf(filename, "%s/%03d.gamma", directory, i);
            save_gamma(filename, var_gamma, c->num_docs);
        }
    }

    // output the final model
    sprintf(filename, "%s/final.model", directory);
    save_model(filename);
    sprintf(filename, "%s/final.model.text", directory);
    save_model_text(filename);
    sprintf(filename, "%s/final.gamma", directory);
    save_gamma(filename, var_gamma, c->num_docs);


    fclose(likelihood_file);
    FILE * w_asgn_file = NULL;
    sprintf(filename, "%s/word-assignments.dat", directory);
    w_asgn_file = fopen(filename, "w");
    for (d = 0; d < c->num_docs; d ++)
    {
        //final inference
        if ((d % 100) == 0) printf("final e step document %d\n", d);
        likelihood += slda_inference(c->docs[d], c->docs_feature[d], var_gamma[d], phi, setting);
        write_word_assignment(w_asgn_file, c->docs[d], phi);

    }
    fclose(w_asgn_file);

    free_suffstats(ss);
    for (d = 0; d < c->num_docs; d++)
        delete [] var_gamma[d];
    delete [] var_gamma;

    for (n = 0; n < max_length; n++)
        delete [] phi[n];
    delete [] phi;
}

//调用：mle(ss, 0, setting);
void slda::mle(suffstats * ss, int eta_update, const settings * setting)
{
    int k, w;

    for (k = 0; k < num_topics; k++)  //对每个主题下，词汇列表的每个词作循环
    {
        for (w = 0; w < size_vocab; w++)
        {
            if (ss->word_ss[k][w] > 0)
                log_prob_w[k][w] = log(ss->word_ss[k][w]) - log(ss->word_total_ss[k]); //Lifeifei：公式10），归一化后取对数
            else
                log_prob_w[k][w] = -100.0; ////the log of the topic distribution
        }
    }

    //初始化主题-特征分布log
    for (k = 0; k < num_topics; k++)  //对每个主题下，词汇列表的每个词作循环
    {
        for (w = 0; w < size_feature; w++)
        {
            if (ss->word_ff[k][w] > 0)
                log_prob_f[k][w] = log(ss->word_ff[k][w]) - log(ss->word_total_ff[k]); //Lifeifei：公式10），归一化后取对数
            else
                log_prob_f[k][w] = -100.0; ////the log of the topic distribution
        }
    }


    if (eta_update == 0) return;

    //the label part goes here
    printf("maximizing ...\n");
	double f = 0.0;
	int status;
	int opt_iter;
	int opt_size = (num_classes-1) * num_topics;  //（分类个数-1）*主题个数
	int l;

    //参数初始化（充分统计量参数、模型和惩罚项）
	opt_parameter param;
	param.ss = ss;
	param.model = this;
	param.PENALTY = setting->PENALTY;

	const gsl_multimin_fdfminimizer_type * T;
	gsl_multimin_fdfminimizer * s;
	gsl_vector * x;
	gsl_multimin_function_fdf opt_fun;
	opt_fun.f = &softmax_f;
	opt_fun.df = &softmax_df;
	opt_fun.fdf = &softmax_fdf;
	opt_fun.n = opt_size;
	opt_fun.params = (void*)(&param);
	x = gsl_vector_alloc(opt_size);


	for (l = 0; l < num_classes-1; l ++)
	{
		for (k = 0; k < num_topics; k ++)
		{
			gsl_vector_set(x, l*num_topics + k, eta[l][k]); //eta:softmax regression,
		}
	}

	T = gsl_multimin_fdfminimizer_vector_bfgs;
	s = gsl_multimin_fdfminimizer_alloc(T, opt_size);               //1>
	gsl_multimin_fdfminimizer_set(s, &opt_fun, x, 0.02, 1e-4);      //2>x:初始点，0.02：step_size，

	opt_iter = 0;
	do
	{
		opt_iter ++;
		status = gsl_multimin_fdfminimizer_iterate(s);
		if (status)
			break;
		status = gsl_multimin_test_gradient(s->gradient, 1e-3);
		if (status == GSL_SUCCESS)
			break; 
		f = -s->f;  //函数负值，最大化MLE转化为最小化问题
		if ((opt_iter-1) % 10 == 0)
			printf("step: %02d -> f: %f\n", opt_iter-1, f); //MLE ：迭代最大化
	} while (status == GSL_CONTINUE && opt_iter < MSTEP_MAX_ITER);

	for (l = 0; l < num_classes-1; l ++)
	{
		for (k = 0; k < num_topics; k ++)
		{
			eta[l][k] = gsl_vector_get(s->x, l*num_topics + k);
		}
	}

	gsl_multimin_fdfminimizer_free (s);
	gsl_vector_free (x);

	printf("final f: %f\n", f);
}
//针对单个文档估计参数：beta
double slda::doc_e_step(document* doc, document *docs_feature, double* gamma, double** phi,
                        suffstats * ss, int eta_update, const settings * setting)
{
    double likelihood = 0.0;
    if (eta_update == 1)//likelihood:即为后验分布
        likelihood = slda_inference(doc, docs_feature, gamma, phi, setting); //推断出变分分布参数，接下来对模型参数作估计
    else
        likelihood = lda_inference(doc, docs_feature, gamma, phi, setting);//LDA推断里，对参数进行初始化

    
    int match[79] = {0,0,1,1,2,2,3,3,3,4,4,5,5,5,6,6,7,7,8,8,9,9,9,10,10,10,10,10,11,11,12,12,12,13,13,13,14,14,14,15,15,16,16,17,17,18,18,19,19,20,20,21,21,21,22,22,23,23,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31,32,32,33,33,33}; // 特征值-特征对应表
    int d = ss->num_docs;

    int n, k, i, idx, m;

    // update sufficient statistics

    for (n = 0; n < doc->length; n++)
        {
            for (k = 0; k < num_topics; k++)
            {
                // ss->word_ss[k][doc->words[n]]：beta参数
                ss->word_ss[k][doc->words[n]] += doc->counts[n]*phi[n][k];//E步计算，在M步估计模型的beta参数，共doc->length * num_topics个
                ss->word_total_ss[k] += doc->counts[n]*phi[n][k];   //每个主题求和，计算beta参数，作参数归一化用

                ss->word_ff[k][match[doc->words[n]]] +=  doc->counts[n] * phi[n][k]; //E步计算，在M步估计模型的beta参数，共doc->length * num_topics个
                ss->word_total_ff[k] += doc->counts[n]*phi[n][k];   //每个主题求和，计算beta参数，作参数归一化用

                
            //statistics for each document of the supervised part
            ss->z_bar[d].z_bar_m[k] += doc->counts[n] * phi[n][k]; //mean
            for (i = k; i < num_topics; i ++) //variance
            {
                idx = map_idx(k, i, num_topics);
                if (i == k)
                    ss->z_bar[d].z_bar_var[idx] +=
                        doc->counts[n] * doc->counts[n] * phi[n][k]; 

                ss->z_bar[d].z_bar_var[idx] -=
                    doc->counts[n] * doc->counts[n] * phi[n][k] * phi[n][i];
            }
        }
    }
    for (k = 0; k < num_topics; k++)
    {
        ss->z_bar[d].z_bar_m[k] /= (double)(doc->total);// 1/N
    }
    for (i = 0; i < num_topics*(num_topics+1)/2; i ++) //  1/N^2
    {
        ss->z_bar[d].z_bar_var[i] /= (double)(doc->total * doc->total);
    }

    ss->num_docs = ss->num_docs + 1; //because we need it for store statistics for each docs

    return (likelihood);
}


//doc：预测的文档
double slda::lda_inference(document* doc,document* docs_feature, double* var_gamma, double** phi, const settings * setting) //预测时，与响应变量无关
{
    int k, n, var_iter;
    double converged = 1, phisum = 0, likelihood = 0, likelihood_old = 0;

    double *oldphi = new double [num_topics];
    double *digamma_gam = new double [num_topics];

    // compute posterior dirichlet
    for (k = 0; k < num_topics; k++)  //变分分布文档-主题分布phi[n][k]: 初始化为 1/K
    {
        var_gamma[k] = alpha + (doc->total/((double) num_topics));  // 公式3）变分参数gamma  N*1/K
        digamma_gam[k] = digamma(var_gamma[k]);  //digamma函数的值：有K个
        for (n = 0; n < doc->length; n++)
            phi[n][k] = 1.0/num_topics;
    }
    
    int match[79] = {0,0,1,1,2,2,3,3,3,4,4,5,5,5,6,6,7,7,8,8,9,9,9,10,10,10,10,10,11,11,12,12,12,13,13,13,14,14,14,15,15,16,16,17,17,18,18,19,19,20,20,21,21,21,22,22,23,23,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31,32,32,33,33,33}; // 特征值-特征对应表

    var_iter = 0;

    while (converged > setting->VAR_CONVERGED && (var_iter < setting->VAR_MAX_ITER || setting->VAR_MAX_ITER == -1))
    {
        var_iter++;
        for (n = 0; n < doc->length; n++) //按不同的词计算
        {
            phisum = 0;
            for (k = 0; k < num_topics; k++) //更新phi:即n个词的变分分布参数
            {
                oldphi[k] = phi[n][k]; 
                phi[n][k] = digamma_gam[k] + log_prob_w[k][doc->words[n]] + log_prob_f[k][match[doc->words[n]]] ; //log_prob_w:主题-词分布的log  slda:公式22）,log_prob_w[k][doc->words[n]]初始化值为：0

                if (k > 0)
                    phisum = log_sum(phisum, phi[n][k]);///log(e^phi[n][0]+...+e^phi[n][k - 1])
                else
                    phisum = phi[n][k]; // note, phi is in log space
            }

            for (k = 0; k < num_topics; k++) //更新gamma
            {
                phi[n][k] = exp(phi[n][k] - phisum); // 对phi[n][k] 作归一化

                var_gamma[k] = var_gamma[k] + doc->counts[n]*(phi[n][k] - oldphi[k]); //第n个词的个数*（变分分布的更新差值）
                digamma_gam[k] = digamma(var_gamma[k]);
            }
        }

        likelihood = lda_compute_likelihood(doc, docs_feature, phi, var_gamma);
        assert(!isnan(likelihood));
        converged = (likelihood_old - likelihood) / likelihood_old;
        likelihood_old = likelihood;
    }

    delete [] oldphi;
    delete [] digamma_gam;

    return likelihood;
}

double slda::lda_compute_likelihood(document* doc, document* docs_feature, double** phi, double* var_gamma)
{
    double likelihood = 0, digsum = 0, var_gamma_sum = 0;
    double *dig = new double [num_topics];
    int k, n;
    double alpha_sum = num_topics * alpha;

    
    int match[79] = {0,0,1,1,2,2,3,3,3,4,4,5,5,5,6,6,7,7,8,8,9,9,9,10,10,10,10,10,11,11,12,12,12,13,13,13,14,14,14,15,15,16,16,17,17,18,18,19,19,20,20,21,21,21,22,22,23,23,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31,32,32,33,33,33}; // 特征值-特征对应表


    for (k = 0; k < num_topics; k++)
    {
        dig[k] = digamma(var_gamma[k]);
        var_gamma_sum += var_gamma[k];
    }
    digsum = digamma(var_gamma_sum);

    likelihood = lgamma(alpha_sum) - lgamma(var_gamma_sum);

    for (k = 0; k < num_topics; k++)
    {
        likelihood += - lgamma(alpha) + (alpha - 1)*(dig[k] - digsum) +
                      lgamma(var_gamma[k]) - (var_gamma[k] - 1)*(dig[k] - digsum);

        for (n = 0; n < doc->length; n++)
        {
            if (phi[n][k] > 0)
            {
                likelihood += doc->counts[n]*(phi[n][k]*((dig[k] - digsum) -
                                              log(phi[n][k]) + log_prob_w[k][doc->words[n]] + log_prob_f[k][match[doc->words[n]]]));
            }
        }
    }

    delete [] dig;
    return likelihood;
}

//根据公式计算ELBO
double slda::slda_compute_likelihood(document* doc, document* docs_feature,double** phi, double* var_gamma)
{
    double likelihood = 0, digsum = 0, var_gamma_sum = 0, t = 0.0, t1 = 0.0, t2 = 0.0;
    double * dig = new double [num_topics];
    int k, n, l;
    int flag;
    double alpha_sum = num_topics * alpha;

    
    int match[79] = {0,0,1,1,2,2,3,3,3,4,4,5,5,5,6,6,7,7,8,8,9,9,9,10,10,10,10,10,11,11,12,12,12,13,13,13,14,14,14,15,15,16,16,17,17,18,18,19,19,20,20,21,21,21,22,22,23,23,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31,32,32,33,33,33}; // 特征值-特征对应表


    for (k = 0; k < num_topics; k++)
    {
        dig[k] = digamma(var_gamma[k]);
        var_gamma_sum += var_gamma[k];
    }
    digsum = digamma(var_gamma_sum);

    likelihood = lgamma(alpha_sum) - lgamma(var_gamma_sum);//sLDA论文中：对应公式6）和公式9）的部分项
    t = 0.0;
    for (k = 0; k < num_topics; k++)
    {
        likelihood += -lgamma(alpha) + (alpha - 1)*(dig[k] - digsum) + lgamma(var_gamma[k]) - (var_gamma[k] - 1)*(dig[k] - digsum);

        for (n = 0; n < doc->length; n++)
        {
            if (phi[n][k] > 0)
            {
                likelihood += doc->counts[n]*(phi[n][k]*((dig[k] - digsum) - log(phi[n][k]) + log_prob_w[k][doc->words[n]] + log_prob_f[k][match[doc->words[n]]]));
                if (doc->label < num_classes-1)
                    t += eta[doc->label][k] * doc->counts[n] * phi[n][k];//sLDA论文中：公式11）第二项，也是Lifeifei公式5）第二项
            }
        }
    }
    likelihood += t / (double)(doc->total); 	//eta_k*\bar{\phi}

    t = 1.0; //the class model->num_classes-1
    for (l = 0; l < num_classes-1; l ++)
    {
        t1 = 1.0; 
        for (n = 0; n < doc->length; n ++)
        {
            t2 = 0.0;
            for (k = 0; k < num_topics; k ++)
            {
                t2 += phi[n][k] * exp(eta[l][k] * doc->counts[n]/(double)(doc->total));//Lifeifei:公式6）
            }
            t1 *= t2; 
        }
        t += t1; 
    }
    likelihood -= log(t); 
    delete [] dig;
    //printf("%lf\n", likelihood);
    return likelihood;
}
// 推断 变分分布参数
double slda::slda_inference(document* doc,document* docs_feature, double* var_gamma, double** phi, const settings * setting)
{
    int k, n, var_iter, l;
    int FP_MAX_ITER = 10;
    int fp_iter = 0;
    double converged = 1, phisum = 0, likelihood = 0, likelihood_old = 0;
    double * oldphi = new double [num_topics];
    double * digamma_gam = new double [num_topics];
    double * sf_params = new double [num_topics];
    double * sf_aux = new double [num_classes-1];
    double sf_val = 0.0;

    // compute posterior dirichlet  对每个主题计算dirichlet后验
    for (k = 0; k < num_topics; k++)
    {
        var_gamma[k] = alpha + (doc->total/((double) num_topics));  //针对单个文档： 初始化文档-主题分布：alpha + 单个文档词语总数/主题个数    
        digamma_gam[k] = digamma(var_gamma[k]);
        for (n = 0; n < doc->length; n++)
            phi[n][k] = 1.0/(double)(num_topics);  //初始化，每个主题下，phi[n]=1/K
    }

    double t = 0.0;
    for (l = 0; l < num_classes-1; l ++)//对每个分类类别作循环
    {
        sf_aux[l] = 1.0; // the quantity for equation 6 of each class
        for (n = 0; n < doc->length; n ++)//doc->length:文档中不同词的个数，论文为词 的个数
        {
            t = 0.0;
            for (k = 0; k < num_topics; k ++)
            {
                t += phi[n][k] * exp(eta[l][k] * doc->counts[n]/(double)(doc->total));//针对不同的词作计算，对应论文6)
            }
            sf_aux[l] *= t;//公式6的计算结果
        }
    }

    var_iter = 0;

    while ((converged > setting->VAR_CONVERGED) && ((var_iter < setting->VAR_MAX_ITER) || (setting->VAR_MAX_ITER == -1)))
    {
        var_iter++;
        for (n = 0; n < doc->length; n++) //对每个不同的词作循环求和，作参数估计
        {
            //compute sf_params
            memset(sf_params, 0, sizeof(double)*num_topics); //in log space
            for (l = 0; l < num_classes-1; l ++)
            {
                t = 0.0;
                for (k = 0; k < num_topics; k ++)
                {
                    t += phi[n][k] * exp(eta[l][k] * doc->counts[n]/(double)(doc->total));
                }
                sf_aux[l] /= t; //take out word n

                for (k = 0; k < num_topics; k ++)
                {
                    //h in the paper
                    sf_params[k] += sf_aux[l]*exp(eta[l][k] * doc->counts[n]/(double)(doc->total));
                }
            }
            //

            
            int match[79] = {0,0,1,1,2,2,3,3,3,4,4,5,5,5,6,6,7,7,8,8,9,9,9,10,10,10,10,10,11,11,12,12,12,13,13,13,14,14,14,15,15,16,16,17,17,18,18,19,19,20,20,21,21,21,22,22,23,23,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31,32,32,33,33,33}; // 特征值-特征对应表
            for (k = 0; k < num_topics; k++)
            {
                oldphi[k] = phi[n][k];
            }
            for (fp_iter = 0; fp_iter < FP_MAX_ITER; fp_iter ++) //fixed point update
            {
                sf_val = 1.0; // the base class, in log space
                for (k = 0; k < num_topics; k++)//log(h*phi_old)
                {
                    sf_val += sf_params[k]*phi[n][k];  // 7)式第四项
                }

                phisum = 0;
                for (k = 0; k < num_topics; k++)
                {
                    phi[n][k] = digamma_gam[k] + log_prob_w[k][doc->words[n]] +log_prob_f[k][match[doc->words[n]]];  //8)式第一项和第二项

                    //added softmax parts
                    if (doc->label < num_classes-1)
                        phi[n][k] += eta[doc->label][k]/(double)(doc->total);// 8)式第三项
                    phi[n][k] -= sf_params[k]/(sf_val*(double)(doc->counts[n]));   //8)式第四项 ，分母除以不同词计数

                    if (k > 0)
                        phisum = log_sum(phisum, phi[n][k]);  //phisum即为8）式log之和，所以8）所有项之和=exp(phisum)
                    else
                        phisum = phi[n][k]; // note, phi is in log space
                }
                for (k = 0; k < num_topics; k++)
                {
                    phi[n][k] = exp(phi[n][k] - phisum); //normalize，即归一化后的8）式，归一化后的推断参数phi
                }
            }
            //back to sf_aux value
            for (l = 0; l < num_classes-1; l ++) //6）式中，每个分类作求和
            {
                t = 0.0;
                for (k = 0; k < num_topics; k ++)
                {
                    t += phi[n][k] * exp(eta[l][k] * doc->counts[n]/(double)(doc->total));
                }
                sf_aux[l] *= t;
            }
            for (k = 0; k < num_topics; k++)
            {
                var_gamma[k] = var_gamma[k] + doc->counts[n]*(phi[n][k] - oldphi[k]);//根据迭代的phi，更新gamma推断参数
                digamma_gam[k] = digamma(var_gamma[k]);
            }
        }

        //通过以上步骤，完成了变分分布参数的推断：phi[n][k]，和var_gamma[k]

        likelihood = slda_compute_likelihood(doc,docs_feature,  phi, var_gamma);//根据每个文档，参数phi和gamma，计算ELBO作为是否结束的判断条件
        assert(!isnan(likelihood));
        converged = fabs((likelihood_old - likelihood) / likelihood_old);
        likelihood_old = likelihood;
    }

    delete [] oldphi;
    delete [] digamma_gam;
    delete [] sf_params;
    delete [] sf_aux;
    return likelihood;
}

void slda::infer_only(corpus * c, const settings * setting, const char * directory)
{
    int i, k, d, n;
    double **var_gamma, likelihood, **phi;
    double* phi_m;
    char filename[100];
    double base_score, score;
    int label;
    int num_correct = 0;
    int max_length = c->max_corpus_length();  //最长文档长度


    var_gamma = new double * [c->num_docs];
    for (i = 0; i < c->num_docs; i++) //文档-主题分布
        var_gamma[i] = new double [num_topics];


    phi = new double * [max_length];
    for (n = 0; n < max_length; n++)//主题-词语分布
        phi[n] = new double [num_topics];

    phi_m = new double [num_topics];

    FILE * accuracy_file = NULL;
    sprintf(filename, "%s/accuracy.dat", ".");
    accuracy_file = fopen(filename, "a");
    FILE * inf_label_file = NULL;
    sprintf(filename, "%s/inf-labels.dat", directory);
    inf_label_file = fopen(filename, "w");

    for (d = 0; d < c->num_docs; d++)
    {
      //  if ((d % 100) == 0)
      //      printf("document %d\n", d);

        document * doc = c->docs[d];
        document * docs_feature = c->docs_feature[d];
        likelihood = lda_inference(doc, docs_feature, var_gamma[d], phi, setting);

        memset(phi_m, 0, sizeof(double)*num_topics); //zero_initialize
        for (n = 0; n < doc->length; n++)
        {
            for (k = 0; k < num_topics; k ++)
            {
                phi_m[k] += doc->counts[n] * phi[n][k];
            }
        }
        for (k = 0; k < num_topics; k ++)
        {
            phi_m[k] /= (double)(doc->total);   // 1/N //推断每个主题下的分布情况的均值
        }

        //do classification
        label = num_classes-1;
        base_score = 0.0;
        for (i = 0; i < num_classes-1; i ++)
        {
            score = 0.0;
            for (k = 0; k < num_topics; k ++)
            {
                score += eta[i][k] * phi_m[k];//计算线性求和的得分 公式13）
            }
            if (score > base_score)  //求出得分最大的标签，也就是预测出来的标签
            {
                base_score = score;
                label = i;
            }
        }
        if (label == doc->label)
            num_correct ++;

        //fprintf(likelihood_file, "%5.5f\n", likelihood);
        fprintf(inf_label_file, "%d,%d\n", doc->label,label);
    }

    printf("average accuracy: %.3f\n", (double)num_correct / (double) c->num_docs);
    fprintf(accuracy_file, "accuracy == %5.5f\n", (double)num_correct / (double) c->num_docs);

    sprintf(filename, "%s/inf-gamma.dat", directory);
    save_gamma(filename, var_gamma, c->num_docs);

    for (d = 0; d < c->num_docs; d++)
        delete [] var_gamma[d];
    delete [] var_gamma;

    for (n = 0; n < max_length; n++)
        delete [] phi[n];
    delete [] phi;

    delete [] phi_m;
}

void slda::save_gamma(char* filename, double** gamma, int num_docs)
{
    int d, k;

    FILE* fileptr = fopen(filename, "w");
    for (d = 0; d < num_docs; d++)
    {
        fprintf(fileptr, "%5.10f", gamma[d][0]);
        for (k = 1; k < num_topics; k++)
            fprintf(fileptr, " %5.10f", gamma[d][k]);
        fprintf(fileptr, "\n");
    }
    fclose(fileptr);
}

void slda::write_word_assignment(FILE* f, document* doc, double** phi)
{
    int n;

    fprintf(f, "%03d", doc->length);  //文档中不同词的个数
    for (n = 0; n < doc->length; n++)
    {
        fprintf(f, " %04d:%02d", doc->words[n], argmax(phi[n], num_topics));
    }
    fprintf(f, "\n");
    fflush(f);
}
