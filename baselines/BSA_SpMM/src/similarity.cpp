#include "similarity.h"

float normalized_weighted_jaccard_sim(vector<intT> A_pattern, vector<intT> B_pattern, intT current_group_size, intT block_size)
{
    float score_A = 0.0;
    float score_B = 0.0;

    float sim_A = 0.0;
    float sim_B = 0.0;

    float min_sum = 0.0;
    float max_sum = 0.0;

    for(int i = 0;i < B_pattern.size();i++)
    {
        score_A += A_pattern[i] * A_pattern[i];
        score_B += B_pattern[i] * B_pattern[i];
    }

    score_A = sqrt(score_A);
    score_B = sqrt(score_B);

    if (score_A == 0 && score_B == 0)
        return 1.0;
    if (score_A == 0 || score_B == 0)
        return 0.0;
    
    for(int i = 0;i < B_pattern.size();i++)
    {
        if (A_pattern[i] == 0 && B_pattern[i] == 0)
        {
            continue;
        }

        sim_A = A_pattern[i] / score_A;
        sim_B = B_pattern[i] / score_B;

        min_sum += min(sim_A, sim_B);
        max_sum += max(sim_A, sim_B);
    }

    return (min_sum / max_sum);
}