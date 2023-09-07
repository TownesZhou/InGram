import torch
from utils import get_rank, get_metrics
from tqdm import tqdm

# Modify to random sampling on both entities and relations
#     1. Randomly sample 50 head entities, 50 tail entities, and 50 relations
#     2. Results for head and tail entities are aggregated 
def evaluate(my_model, target, epoch, init_emb_ent, init_emb_rel, relation_triplets):
    with torch.no_grad():
        my_model.eval()
        msg = torch.tensor(target.msg_triplets).cuda()
        sup = torch.tensor(target.sup_triplets).cuda()

        emb_ent, emb_rel = my_model(init_emb_ent, init_emb_rel, msg, relation_triplets)

        ent_ranks = []
        rel_ranks = []
        for triplet in tqdm(sup):
            
            triplet = triplet.unsqueeze(dim = 0)

            ### ORIGINAL CODE ###
            # head_corrupt = triplet.repeat(target.num_ent, 3)
            # head_corrupt[:,0] = torch.arange(end = target.num_ent)
            
            # head_scores = my_model.score(emb_ent, emb_rel, head_corrupt)
            # head_filters = target.filter_dict[('_', int(triplet[0,1].item()), int(triplet[0,2].item()))]
            # head_rank = get_rank(triplet, head_scores, head_filters, target = 0)

            ### MODIFY: Random sample 50 corrupt heads instead of all ###
            head_corrupt = triplet.repeat(51, 3)   # 51 because 50 + 1 (for the original triplet)
            # head_filters gives us the other positive triplets with the same relation and tail
            head_filters = target.filter_dict[('_', int(triplet[0,1].item()), int(triplet[0,2].item()))]
            # Sample 50 corrupt heads that we know are not in the filter
            neg_heads_all = torch.tensor(list(set(range(target.num_ent)) - set(head_filters)))
            neg_heads_sampled_ids = torch.randint(low=0, high = len(neg_heads_all)-1, size = (50,))
            head_corrupt[1:,0] = neg_heads_all[neg_heads_sampled_ids]
            # Compute the score            
            head_scores_51 = my_model.score(emb_ent, emb_rel, head_corrupt)
            
            # The original head_scores has score for every head, and get_rank() can only work with that
            # So to work around this, create a head_scores tensor with the same length as the number of entities
            # And for the original head and those sampled entities, fill in the scores from head_scores_51
            # For the rest of the entities, fill in a score of original head score - 1, that is, head_scores_51[0] - 1
            head_scores = torch.ones(target.num_ent).cuda() * (head_scores_51[0] - 1)
            head_scores[head_corrupt[:,0]] = head_scores_51
            head_rank = get_rank(triplet, head_scores, head_filters, target = 0)

            ### ORIGINAL CODE ###
            # tail_corrupt = triplet.repeat(target.num_ent, 3)
            # tail_corrupt[:,2] = torch.arange(end = target.num_ent)

            # tail_scores = my_model.score(emb_ent, emb_rel, tail_corrupt)
            # tail_filters = target.filter_dict[(int(triplet[0,0].item()), int(triplet[0,1].item()), '_')]
            # tail_rank = get_rank(triplet, tail_scores, tail_filters, target = 2)

            ### MODIFY: Random sample 50 corrupt tails instead of all ###
            tail_corrupt = triplet.repeat(51, 3)
            tail_filters = target.filter_dict[(int(triplet[0,0].item()), int(triplet[0,1].item()), '_')]
            neg_tails_all = torch.tensor(list(set(range(target.num_ent)) - set(tail_filters)))
            neg_tails_sampled_ids = torch.randint(low=0, high = len(neg_tails_all)-1, size = (50,))
            tail_corrupt[1:,2] = neg_tails_all[neg_tails_sampled_ids]

            tail_scores_51 = my_model.score(emb_ent, emb_rel, tail_corrupt)

            tail_scores = torch.ones(target.num_ent).cuda() * (tail_scores_51[0] - 1)
            tail_scores[tail_corrupt[:,2]] = tail_scores_51
            tail_rank = get_rank(triplet, tail_scores, tail_filters, target = 2)

            ### MODIFY: Random sample 50 corrupt relations###
            rel_corrupt = triplet.repeat(51, 3)
            rel_filters = target.filter_dict[(int(triplet[0,0].item()), '_', int(triplet[0,2].item()))]
            neg_rels_all = torch.tensor(list(set(range(target.num_rel)) - set(rel_filters)))
            neg_rels_sampled_ids = torch.randint(low=0, high = len(neg_rels_all)-1, size = (50,))
            rel_corrupt[1:,1] = neg_rels_all[neg_rels_sampled_ids]

            rel_scores_51 = my_model.score(emb_ent, emb_rel, rel_corrupt)

            rel_scores = torch.ones(target.num_rel).cuda() * (rel_scores_51[0] - 1)
            rel_scores[rel_corrupt[:,1]] = rel_scores_51
            rel_rank = get_rank(triplet, rel_scores, rel_filters, target = 1)

            ent_ranks.append(head_rank)
            ent_ranks.append(tail_rank)
            rel_ranks.append(rel_rank)

        print("--------LP--------")
        mr_ent, mrr_ent, hit10_ent, hit3_ent, hit1_ent = get_metrics(ent_ranks)
        mr_rel, mrr_rel, hit10_rel, hit3_rel, hit1_rel = get_metrics(rel_ranks)
        print(f"MR_ENT: {mr_ent:.1f}")
        print(f"MRR_ENT: {mrr_ent:.3f}")
        print(f"Hits@10_ENT: {hit10_ent:.3f}")
        print(f"Hits@1_ENT: {hit1_ent:.3f}")
        print("--------")
        print(f"MR_REL: {mr_rel:.1f}")
        print(f"MRR_REL: {mrr_rel:.3f}")
        print(f"Hits@10_REL: {hit10_rel:.3f}")
        print(f"Hits@1_REL: {hit1_rel:.3f}")

        # Return the metrics in dict
        return {
            'mr_ent': mr_ent, 'mrr_ent': mrr_ent, 'hit10_ent': hit10_ent, 'hit3_ent': hit3_ent, 'hit1_ent': hit1_ent,
            'mr_rel': mr_rel, 'mrr_rel': mrr_rel, 'hit10_rel': hit10_rel, 'hit3_rel': hit3_rel, 'hit1_rel': hit1_rel}
    