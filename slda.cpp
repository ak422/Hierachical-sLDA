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
    alpha = alpha_;                 //������
    num_topics = num_topics_;       //�������
    size_vocab = c->size_vocab;     //����ֵ�б���
    size_feature = c->size_feature;     //�����б���
    num_classes = c->num_classes;  //������Ŀ

    log_prob_w = new double * [num_topics]; //the log of the topic distribution
    for (int k = 0; k < num_topics; k++)
    {
        log_prob_w[k] = new double [size_vocab];  //ÿһ������ֲ������������ʻ��б���
        memset(log_prob_w[k], 0, sizeof(double)*size_vocab);  //��ʼ��Ϊ0
    }

    // by ak422
    log_prob_f = new double * [num_topics]; //the log of the topic distribution
    for (int k = 0; k < num_topics; k++)
    {
        log_prob_f[k] = new double [size_feature];  //ÿһ������ֲ������������ʻ��б���
        memset(log_prob_f[k], 0, sizeof(double)*size_feature);  //��ʼ��Ϊ0
    }


    //no need to train slda if we only have one class
    if (num_classes > 1)
    {
        eta = new double * [num_classes-1];  //num_classes-1 �������eta����
        for (int i = 0; i < num_classes-1; i ++)
        {
            eta[i] = new double [num_topics];  //ÿ�����඼��������������һ��ϵ��
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

    //��ȡģ���� ����-���ʷֲ���log
    log_prob_w = new double * [num_topics];
    for (int k = 0; k < num_topics; k++)
    {
        log_prob_w[k] = new double [size_vocab];
        fread(log_prob_w[k], sizeof(double), size_vocab, file);
    }


    //��ȡģ���� ����-�����ֲ���log
    log_prob_f = new double * [num_topics];
    for (int k = 0; k < num_topics; k++)
    {
        log_prob_f[k] = new double [size_feature];
        fread(log_prob_f[k], sizeof(double), size_feature, file);
    }


    //��ȡģ���� �����softmax�ع�ϵ��
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
//���ͳ����
suffstats * slda::new_suffstats(int num_docs)
{
    suffstats * ss = new suffstats;
    ss->num_docs = num_docs;        //�ĵ�����

    ss->word_total_ss = new double [num_topics];  //ÿ�����������ֵͳ��
    memset(ss->word_total_ss, 0, sizeof(double)*num_topics);  //ÿ�����������ֵͳ�Ƴ�ʼ��Ϊ0
    
    ss->word_ss = new double * [num_topics];  //ÿ�������µĴʻ��ķֲ�
    for (int k = 0; k < num_topics; k ++)
    {
        ss->word_ss[k] = new double [size_vocab];
        memset(ss->word_ss[k], 0, sizeof(double)*size_vocab);  //ÿ������-����ֵ�ֲ���ʼ��Ϊ0 
    }


    ss->word_total_ff = new double [num_topics];  //ÿ�����������ͳ��
    memset(ss->word_total_ff, 0, sizeof(double)*num_topics);  //ÿ�����������ͳ�Ƴ�ʼ��Ϊ0
   
    ss->word_ff = new double * [num_topics];  //ÿ�������µĴʻ��ķֲ�
    for (int k = 0; k < num_topics; k ++)
    {
        ss->word_ff[k] = new double [size_feature];
        memset(ss->word_ff[k], 0, sizeof(double)*size_feature);  //ÿ������ֲ�-������ʼ��Ϊ0 
    }



    int num_var_entries = num_topics*(num_topics+1)/2;
    ss->z_bar =  new z_stat [num_docs];
    for (int d = 0; d < num_docs; d ++)
    {
        ss->z_bar[d].z_bar_m = new double [num_topics];  //ÿ���ĵ��µ�����ֲ�
        ss->z_bar[d].z_bar_var = new double [num_var_entries];  //ÿ���ĵ���num_topics*(num_topics+1)/2��
        memset(ss->z_bar[d].z_bar_m, 0, sizeof(double)*num_topics);
        memset(ss->z_bar[d].z_bar_var, 0, sizeof(double)*num_var_entries);
    }
    ss->labels = new int [num_docs];   //ÿ���ĵ�һ����ǩ
    memset(ss->labels, 0, sizeof(int)*(num_docs));
    ss->tot_labels = new int [num_classes];  //����ǩ����ͳ��
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



// �����ʼ������-�ʻ��ֲ������ĵ�-����ֲ�
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
            ss->word_total_ss[k] += ss->word_ss[k][w];  //word_ss[k][w]������-�ʻ��ֲ�
        }

        for (w = 0; w < size_feature; w++)
        {
            ss->word_ff[k][w] = 1.0/size_feature + 0.1*gsl_rng_uniform(rng);
            ss->word_total_ff[k] += ss->word_ff[k][w];  //word_ss[k][w]������-�ʻ��ֲ�
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


// �������Ͽ⣬����ʵ���ݳ�ʼ������-�ʻ��ֲ������ĵ�-����ֲ���δ�������ĵ�ɨ�裬��һ������ֵ��
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
        for (i = 0; i < NUM_INIT; i++)  //���ѡ��ĳ���ĵ������Ĵ�������NUM_INIT�������ȡ��ÿ������NUM_INIT=50�����ѡ���ĵ����㣩
        {
            d = (int)(floor(gsl_rng_uniform(rng) * num_docs)); //����һ����[0, 1)�����ϵ�˫����������������������������0����������1��
            printf("initialized with document %d\n", d);//���ѡ��һ���ĵ�

            document * doc = c->docs[d];
            for (n = 0; n < doc->length; n++)
            {
                //doc->words[n]���ĵ���n�����Ǵʻ��б�ĵڼ�����
                //ss->word_ss[k][doc->words[n]]����k�����⣬��doc->words[n]�ʵ�����
                ss->word_ss[k][doc->words[n]] += doc->counts[n]; //ͳ���ĵ���ÿ��������-���ʻ��б�ͳ�����

            }


            document * docs_feature = c->docs_feature[d];
            for (m = 0; m < docs_feature->length; m++)
            {
                //doc->words[n]���ĵ���n�����Ǵʻ��б�ĵڼ�����
                //ss->word_ss[k][doc->words[n]]����k�����⣬��doc->words[n]�ʵ�����
                ss->word_ff[k][docs_feature->words[m]] += docs_feature->counts[m]; //ͳ���ĵ���ÿ��������-���ʻ��б�ͳ�����
            }            

        }

        for (w = 0; w < size_vocab; w++) //ÿ�������£��ʻ��б��е�ÿ����
        {
            ss->word_ss[k][w] = 2*ss->word_ss[k][w] + 5 + gsl_rng_uniform(rng); // by removing words that occur in fewer than 5 documents.
            ss->word_total_ss[k] = ss->word_total_ss[k] + ss->word_ss[k][w]; //ͳ��ÿ�������µĴʸ���
        }


        for (w = 0; w < size_feature; w++) //ÿ�������£��ʻ��б��е�ÿ����
        {
            ss->word_ff[k][w] = 2*ss->word_ff[k][w] + 5 + gsl_rng_uniform(rng); // by removing words that occur in fewer than 5 documents.
            ss->word_total_ff[k] = ss->word_total_ff[k] + ss->word_ff[k][w]; //ͳ��ÿ�������µĴʸ���
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
    for (int d = 0; d < num_docs; d ++)   //�������Ͽ��ʼ�����ͳ�����ı�ǩ    
    {                                                                                                    
       document * doc = c->docs[d];     //ÿ���ĵ�һ����ǩ
       ss->labels[d] = doc->label;
       ss->tot_labels[doc->label] ++;   //ͳ��ÿ����ǩ���������ĵ�����ÿ����ǩ�ļ���
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
    int max_length = c->max_corpus_length(); //�����ĵ�������ָ��ͬ������������ֵ
    double **var_gamma, **phi, **lambda;
    double likelihood, likelihood_old = 0, converged = 1;
    int d, n, i;
    double L2penalty = setting->PENALTY;
    // allocate variational parameters
    var_gamma = new double * [c->num_docs];   //�ĵ����� ��ÿһ���ĵ�����һ������ֲ�
    for (d = 0; d < c->num_docs; d++)
        var_gamma[d] = new double [num_topics];     //�ĵ�-����ֲ�����ֲ���

    phi = new double * [max_length];  //ÿһ�����ʶ���һ������ֲ�
    for (n = 0; n < max_length; n++)
        phi[n] = new double [num_topics];//����ֲ���

    printf("initializing ...\n");
    suffstats * ss = new_suffstats(c->num_docs);//��ʼ�����ͳ���������ݽṹ
    if (strcmp(start, "seeded") == 0)  //seeded/random/model_path
    {
        corpus_initialize_ss(ss, c);
        mle(ss, 0, setting);//����������ʼ��
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
        for (d = 0; d < c->num_docs; d++)  //��ÿ���ĵ�ִ��E-STEP
        {
            if ((d % 100) == 0) printf("document %d\n", d);
            likelihood += doc_e_step(c->docs[d],c->docs_feature[d], var_gamma[d], phi, ss, ETA_UPDATE, setting); //ETA_UPDATE =1 :sLDA �ƶϣ�ETA_UPDATE =0 :LDA �ƶ�
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

//���ã�mle(ss, 0, setting);
void slda::mle(suffstats * ss, int eta_update, const settings * setting)
{
    int k, w;

    for (k = 0; k < num_topics; k++)  //��ÿ�������£��ʻ��б��ÿ������ѭ��
    {
        for (w = 0; w < size_vocab; w++)
        {
            if (ss->word_ss[k][w] > 0)
                log_prob_w[k][w] = log(ss->word_ss[k][w]) - log(ss->word_total_ss[k]); //Lifeifei����ʽ10������һ����ȡ����
            else
                log_prob_w[k][w] = -100.0; ////the log of the topic distribution
        }
    }

    //��ʼ������-�����ֲ�log
    for (k = 0; k < num_topics; k++)  //��ÿ�������£��ʻ��б��ÿ������ѭ��
    {
        for (w = 0; w < size_feature; w++)
        {
            if (ss->word_ff[k][w] > 0)
                log_prob_f[k][w] = log(ss->word_ff[k][w]) - log(ss->word_total_ff[k]); //Lifeifei����ʽ10������һ����ȡ����
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
	int opt_size = (num_classes-1) * num_topics;  //���������-1��*�������
	int l;

    //������ʼ�������ͳ����������ģ�ͺͳͷ��
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
	gsl_multimin_fdfminimizer_set(s, &opt_fun, x, 0.02, 1e-4);      //2>x:��ʼ�㣬0.02��step_size��

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
		f = -s->f;  //������ֵ�����MLEת��Ϊ��С������
		if ((opt_iter-1) % 10 == 0)
			printf("step: %02d -> f: %f\n", opt_iter-1, f); //MLE ���������
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
//��Ե����ĵ����Ʋ�����beta
double slda::doc_e_step(document* doc, document *docs_feature, double* gamma, double** phi,
                        suffstats * ss, int eta_update, const settings * setting)
{
    double likelihood = 0.0;
    if (eta_update == 1)//likelihood:��Ϊ����ֲ�
        likelihood = slda_inference(doc, docs_feature, gamma, phi, setting); //�ƶϳ���ֲַ���������������ģ�Ͳ���������
    else
        likelihood = lda_inference(doc, docs_feature, gamma, phi, setting);//LDA�ƶ���Բ������г�ʼ��

    
    int match[79] = {0,0,1,1,2,2,3,3,3,4,4,5,5,5,6,6,7,7,8,8,9,9,9,10,10,10,10,10,11,11,12,12,12,13,13,13,14,14,14,15,15,16,16,17,17,18,18,19,19,20,20,21,21,21,22,22,23,23,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31,32,32,33,33,33}; // ����ֵ-������Ӧ��
    int d = ss->num_docs;

    int n, k, i, idx, m;

    // update sufficient statistics

    for (n = 0; n < doc->length; n++)
        {
            for (k = 0; k < num_topics; k++)
            {
                // ss->word_ss[k][doc->words[n]]��beta����
                ss->word_ss[k][doc->words[n]] += doc->counts[n]*phi[n][k];//E�����㣬��M������ģ�͵�beta��������doc->length * num_topics��
                ss->word_total_ss[k] += doc->counts[n]*phi[n][k];   //ÿ��������ͣ�����beta��������������һ����

                ss->word_ff[k][match[doc->words[n]]] +=  doc->counts[n] * phi[n][k]; //E�����㣬��M������ģ�͵�beta��������doc->length * num_topics��
                ss->word_total_ff[k] += doc->counts[n]*phi[n][k];   //ÿ��������ͣ�����beta��������������һ����

                
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


//doc��Ԥ����ĵ�
double slda::lda_inference(document* doc,document* docs_feature, double* var_gamma, double** phi, const settings * setting) //Ԥ��ʱ������Ӧ�����޹�
{
    int k, n, var_iter;
    double converged = 1, phisum = 0, likelihood = 0, likelihood_old = 0;

    double *oldphi = new double [num_topics];
    double *digamma_gam = new double [num_topics];

    // compute posterior dirichlet
    for (k = 0; k < num_topics; k++)  //��ֲַ��ĵ�-����ֲ�phi[n][k]: ��ʼ��Ϊ 1/K
    {
        var_gamma[k] = alpha + (doc->total/((double) num_topics));  // ��ʽ3����ֲ���gamma  N*1/K
        digamma_gam[k] = digamma(var_gamma[k]);  //digamma������ֵ����K��
        for (n = 0; n < doc->length; n++)
            phi[n][k] = 1.0/num_topics;
    }
    
    int match[79] = {0,0,1,1,2,2,3,3,3,4,4,5,5,5,6,6,7,7,8,8,9,9,9,10,10,10,10,10,11,11,12,12,12,13,13,13,14,14,14,15,15,16,16,17,17,18,18,19,19,20,20,21,21,21,22,22,23,23,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31,32,32,33,33,33}; // ����ֵ-������Ӧ��

    var_iter = 0;

    while (converged > setting->VAR_CONVERGED && (var_iter < setting->VAR_MAX_ITER || setting->VAR_MAX_ITER == -1))
    {
        var_iter++;
        for (n = 0; n < doc->length; n++) //����ͬ�Ĵʼ���
        {
            phisum = 0;
            for (k = 0; k < num_topics; k++) //����phi:��n���ʵı�ֲַ�����
            {
                oldphi[k] = phi[n][k]; 
                phi[n][k] = digamma_gam[k] + log_prob_w[k][doc->words[n]] + log_prob_f[k][match[doc->words[n]]] ; //log_prob_w:����-�ʷֲ���log  slda:��ʽ22��,log_prob_w[k][doc->words[n]]��ʼ��ֵΪ��0

                if (k > 0)
                    phisum = log_sum(phisum, phi[n][k]);///log(e^phi[n][0]+...+e^phi[n][k - 1])
                else
                    phisum = phi[n][k]; // note, phi is in log space
            }

            for (k = 0; k < num_topics; k++) //����gamma
            {
                phi[n][k] = exp(phi[n][k] - phisum); // ��phi[n][k] ����һ��

                var_gamma[k] = var_gamma[k] + doc->counts[n]*(phi[n][k] - oldphi[k]); //��n���ʵĸ���*����ֲַ��ĸ��²�ֵ��
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

    
    int match[79] = {0,0,1,1,2,2,3,3,3,4,4,5,5,5,6,6,7,7,8,8,9,9,9,10,10,10,10,10,11,11,12,12,12,13,13,13,14,14,14,15,15,16,16,17,17,18,18,19,19,20,20,21,21,21,22,22,23,23,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31,32,32,33,33,33}; // ����ֵ-������Ӧ��


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

//���ݹ�ʽ����ELBO
double slda::slda_compute_likelihood(document* doc, document* docs_feature,double** phi, double* var_gamma)
{
    double likelihood = 0, digsum = 0, var_gamma_sum = 0, t = 0.0, t1 = 0.0, t2 = 0.0;
    double * dig = new double [num_topics];
    int k, n, l;
    int flag;
    double alpha_sum = num_topics * alpha;

    
    int match[79] = {0,0,1,1,2,2,3,3,3,4,4,5,5,5,6,6,7,7,8,8,9,9,9,10,10,10,10,10,11,11,12,12,12,13,13,13,14,14,14,15,15,16,16,17,17,18,18,19,19,20,20,21,21,21,22,22,23,23,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31,32,32,33,33,33}; // ����ֵ-������Ӧ��


    for (k = 0; k < num_topics; k++)
    {
        dig[k] = digamma(var_gamma[k]);
        var_gamma_sum += var_gamma[k];
    }
    digsum = digamma(var_gamma_sum);

    likelihood = lgamma(alpha_sum) - lgamma(var_gamma_sum);//sLDA�����У���Ӧ��ʽ6���͹�ʽ9���Ĳ�����
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
                    t += eta[doc->label][k] * doc->counts[n] * phi[n][k];//sLDA�����У���ʽ11���ڶ��Ҳ��Lifeifei��ʽ5���ڶ���
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
                t2 += phi[n][k] * exp(eta[l][k] * doc->counts[n]/(double)(doc->total));//Lifeifei:��ʽ6��
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
// �ƶ� ��ֲַ�����
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

    // compute posterior dirichlet  ��ÿ���������dirichlet����
    for (k = 0; k < num_topics; k++)
    {
        var_gamma[k] = alpha + (doc->total/((double) num_topics));  //��Ե����ĵ��� ��ʼ���ĵ�-����ֲ���alpha + �����ĵ���������/�������    
        digamma_gam[k] = digamma(var_gamma[k]);
        for (n = 0; n < doc->length; n++)
            phi[n][k] = 1.0/(double)(num_topics);  //��ʼ����ÿ�������£�phi[n]=1/K
    }

    double t = 0.0;
    for (l = 0; l < num_classes-1; l ++)//��ÿ�����������ѭ��
    {
        sf_aux[l] = 1.0; // the quantity for equation 6 of each class
        for (n = 0; n < doc->length; n ++)//doc->length:�ĵ��в�ͬ�ʵĸ���������Ϊ�� �ĸ���
        {
            t = 0.0;
            for (k = 0; k < num_topics; k ++)
            {
                t += phi[n][k] * exp(eta[l][k] * doc->counts[n]/(double)(doc->total));//��Բ�ͬ�Ĵ������㣬��Ӧ����6)
            }
            sf_aux[l] *= t;//��ʽ6�ļ�����
        }
    }

    var_iter = 0;

    while ((converged > setting->VAR_CONVERGED) && ((var_iter < setting->VAR_MAX_ITER) || (setting->VAR_MAX_ITER == -1)))
    {
        var_iter++;
        for (n = 0; n < doc->length; n++) //��ÿ����ͬ�Ĵ���ѭ����ͣ�����������
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

            
            int match[79] = {0,0,1,1,2,2,3,3,3,4,4,5,5,5,6,6,7,7,8,8,9,9,9,10,10,10,10,10,11,11,12,12,12,13,13,13,14,14,14,15,15,16,16,17,17,18,18,19,19,20,20,21,21,21,22,22,23,23,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31,32,32,33,33,33}; // ����ֵ-������Ӧ��
            for (k = 0; k < num_topics; k++)
            {
                oldphi[k] = phi[n][k];
            }
            for (fp_iter = 0; fp_iter < FP_MAX_ITER; fp_iter ++) //fixed point update
            {
                sf_val = 1.0; // the base class, in log space
                for (k = 0; k < num_topics; k++)//log(h*phi_old)
                {
                    sf_val += sf_params[k]*phi[n][k];  // 7)ʽ������
                }

                phisum = 0;
                for (k = 0; k < num_topics; k++)
                {
                    phi[n][k] = digamma_gam[k] + log_prob_w[k][doc->words[n]] +log_prob_f[k][match[doc->words[n]]];  //8)ʽ��һ��͵ڶ���

                    //added softmax parts
                    if (doc->label < num_classes-1)
                        phi[n][k] += eta[doc->label][k]/(double)(doc->total);// 8)ʽ������
                    phi[n][k] -= sf_params[k]/(sf_val*(double)(doc->counts[n]));   //8)ʽ������ ����ĸ���Բ�ͬ�ʼ���

                    if (k > 0)
                        phisum = log_sum(phisum, phi[n][k]);  //phisum��Ϊ8��ʽlog֮�ͣ�����8��������֮��=exp(phisum)
                    else
                        phisum = phi[n][k]; // note, phi is in log space
                }
                for (k = 0; k < num_topics; k++)
                {
                    phi[n][k] = exp(phi[n][k] - phisum); //normalize������һ�����8��ʽ����һ������ƶϲ���phi
                }
            }
            //back to sf_aux value
            for (l = 0; l < num_classes-1; l ++) //6��ʽ�У�ÿ�����������
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
                var_gamma[k] = var_gamma[k] + doc->counts[n]*(phi[n][k] - oldphi[k]);//���ݵ�����phi������gamma�ƶϲ���
                digamma_gam[k] = digamma(var_gamma[k]);
            }
        }

        //ͨ�����ϲ��裬����˱�ֲַ��������ƶϣ�phi[n][k]����var_gamma[k]

        likelihood = slda_compute_likelihood(doc,docs_feature,  phi, var_gamma);//����ÿ���ĵ�������phi��gamma������ELBO��Ϊ�Ƿ�������ж�����
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
    int max_length = c->max_corpus_length();  //��ĵ�����


    var_gamma = new double * [c->num_docs];
    for (i = 0; i < c->num_docs; i++) //�ĵ�-����ֲ�
        var_gamma[i] = new double [num_topics];


    phi = new double * [max_length];
    for (n = 0; n < max_length; n++)//����-����ֲ�
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
            phi_m[k] /= (double)(doc->total);   // 1/N //�ƶ�ÿ�������µķֲ�����ľ�ֵ
        }

        //do classification
        label = num_classes-1;
        base_score = 0.0;
        for (i = 0; i < num_classes-1; i ++)
        {
            score = 0.0;
            for (k = 0; k < num_topics; k ++)
            {
                score += eta[i][k] * phi_m[k];//����������͵ĵ÷� ��ʽ13��
            }
            if (score > base_score)  //����÷����ı�ǩ��Ҳ����Ԥ������ı�ǩ
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

    fprintf(f, "%03d", doc->length);  //�ĵ��в�ͬ�ʵĸ���
    for (n = 0; n < doc->length; n++)
    {
        fprintf(f, " %04d:%02d", doc->words[n], argmax(phi[n], num_topics));
    }
    fprintf(f, "\n");
    fflush(f);
}
