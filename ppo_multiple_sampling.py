from paroutes import PaRoutesInventory, PaRoutesModel, get_target_smiles
from GNN_utils import *
from split_environment import *
from tqdm import tqdm
import pickle

clipping_val = 0.2
critic_discount = 0.5
entropy_beta = 0.001
gamma = 0.99
lmbda = 0.95


def ppo_loss(new_preds, old_preds, adv, rewards, values):
    # print(new_preds,old_preds,adv,rewards,values)
    ratio = torch.exp(torch.log(new_preds + 1e-10) - torch.log(old_preds + 1e-10))
    p1 = ratio * adv
    p2 = torch.clip(ratio, min=1 - clipping_val, max=1 + clipping_val) * adv
    act_loss = -torch.mean(torch.min(p1, p2))
    crt_loss = torch.mean(torch.square(rewards - values))
    tot_loss = (
        critic_discount * crt_loss
        + act_loss
        - entropy_beta * torch.mean(-(new_preds * torch.log(new_preds + 1e-10)))
    )
    return tot_loss


def get_advantages(values, masks, rewards):
    returns = torch.empty(0)

    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae

        res = gae + values[i]
        # try:
        returns = torch.cat((res.reshape(1), returns))
        # except:
        # returns = torch.cat((returns,gae+values[i]))
        # returns.insert(0,gae+values[i])

    adv = returns - values[:-1]
    return (returns - torch.min(returns)) / (torch.max(returns) + 1e-10), (
        adv - torch.mean(adv)
    ) / (torch.std(adv) + 1e-10)


def test_reward(dataset, env, model_actor):
    model_actor.eval()
    done = False
    tot_reward = 0
    print("testing...")
    limit = 0
    cnt = 2500
    REPEAT=5
    state_input = dataset[cnt]

    while not done:
        inv = env.get_inventory()
        cum_reward=[]
        mxm_rew = 0
        best_act = -1
        for i in range(REPEAT):
            try:
                action_probs = model_actor(convert_smiles_to_pyg_graph(state_input))
            except:
                state_input = dataset[cnt + 1]
                cnt = (cnt + 1) % len(dataset)
                continue
            action_probs_numpy = action_probs.detach().numpy()
            action = np.random.choice(len(action_probs_numpy),p=action_probs_numpy)
            bond = get_BRICS_bonds(Chem.MolFromSmiles(state_input))

            bond = [list(i[0]) for i in bond]
            bond.sort()
            bond = bond[action]
            mols = break_bond(Chem.MolFromSmiles(state_input),[bond])
            est_rew = calculate_estimate(mols,inv)
            cum_reward.append(est_rew)
            if est_rew > mxm_rew:
                best_act = action
                mxm_rew = est_rew

        bond = get_BRICS_bonds(Chem.MolFromSmiles(state_input))

        bond = [list(i[0]) for i in bond]
        bond.sort()
        bond = bond[best_act]
        observation,reward,done = env.step([bond],state_input)
        print(cum_reward,reward)
        state_input = dataset[cnt + 1]
        cnt = (cnt + 1) % len(dataset)
        tot_reward += reward
        limit += 1
        if limit > 20:
            break
    print(cnt)
    return tot_reward


def main():
    ppo_steps = 128
    target_reached = False
    best_reward = 0
    iters = 0
    EPOCHS = 15
    torch.autograd.set_detect_anomaly(True)
    max_iters = 100

    dataset = get_target_smiles(5)
    cnt = 0
    REPEAT=5

    model_actor = GAT(9, 32)
    model_critic = GATCritic(9, 1)
    env = SASEnvironment(dataset)
    optimizer_act = torch.optim.Adam(model_actor.parameters(), lr=1e-4)
    optimizer_critic = torch.optim.Adam(model_critic.parameters(), lr=1e-4)
    avg_reward_list = []
    while not target_reached and iters < max_iters:
        states = []
        actions = []
        values = torch.empty(0)
        masks = []
        rewards = []
        actions_probs = []
        actions_onehot = []
        state_input = dataset[cnt]

        for itr in range(ppo_steps):
            # state_input = K.expand_dims(state, 0)

            brics = get_BRICS_bonds(Chem.MolFromSmiles(state_input))
            if brics:
                tot_reward = []
                for i in range(REPEAT):
                    action_dist = model_actor(convert_smiles_to_pyg_graph(state_input))
                
                    q_value = model_critic(convert_smiles_to_pyg_graph(state_input))
                    action_dist_numpy = action_dist.detach().numpy()
                    # print(action_dist_numpy)
                    action = np.random.choice(
                        range(len(action_dist_numpy)),p=action_dist_numpy
                    )
                    action_onehot = np.zeros(len(action_dist_numpy))
                    action_onehot[action] = 1

                    bond = get_BRICS_bonds(Chem.MolFromSmiles(state_input))

                    bond = [list(i[0]) for i in bond]
                    bond.sort()

                    # print(action,len(bond))

                    bond = bond[action]

                    observation, reward, done = env.step([bond], state_input)
                    # print('itr: ' + str(itr) + ', action=' + str(action) + ', reward=' + str(reward) + ', q val=' + str(q_value))
                    mask = not done

                    states.append(state_input)
                    actions.append(action)
                    actions_onehot.append(action_onehot)
                    values = torch.cat((values.clone(), q_value))
                    # values.append(q_value)
                    # masks = torch.cat((masks,torch.tensormask))
                    masks.append(mask)
                    # rewards = torch.cat((rewards,reward))
                    rewards.append(reward)
                    tot_reward.append(reward)
                    actions_probs.append(action_dist)
                print(tot_reward,max(tot_reward))
                state_input = dataset[cnt+1]
                cnt = (cnt+1)%1000

            if done:
                state_input = dataset[cnt + 1]
                cnt = (cnt + 1) % 1000

        q_value = model_critic(convert_smiles_to_pyg_graph(state_input))
        values = torch.cat((values, q_value))
        # print(values,returns)
        returns, advantages = get_advantages(values, masks, rewards)
        # print(returns,advantages)

        # print(values,returns)

        print("Training Actor")

        for epoch in range(EPOCHS):
            model_actor.train()
            for i, state in enumerate(states):
                with torch.autograd.set_detect_anomaly(True):
                    new_preds = model_actor(convert_smiles_to_pyg_graph(state))
                    loss = ppo_loss(
                        new_preds,
                        actions_probs[i].detach(),
                        advantages[i].detach(),
                        rewards[i],
                        values[i].detach(),
                    )
                    # print(loss)
                    optimizer_act.zero_grad()
                    loss.backward()
                    optimizer_act.step()
            print(f"{epoch}/{EPOCHS} DONE", loss.item())

        print("Training Critic")
        for epoch in range(EPOCHS):
            model_critic.train()
            for i, state in enumerate(states):
                crt_pred = model_critic(convert_smiles_to_pyg_graph(state))
                crt_loss = torch.square(crt_pred - returns[i].detach())
                optimizer_critic.zero_grad()
                crt_loss.backward()
                optimizer_critic.step()
            print(f"{epoch}/{EPOCHS} Done", crt_loss.item())
        print("Iteration", iters)
        avg_reward = np.mean([test_reward(dataset, env, model_actor) for i in range(5)])
        print("TEST AVG REWARD:", avg_reward)
        avg_reward_list.append(avg_reward)
        # x=input()
        iters += 1
        with open("avg_reward", "wb") as fp:
            pickle.dump(avg_reward_list, fp)
        torch.save(model_actor, "GNN_split_Multi_Sample_NN_Model.pth")


if __name__ == "__main__":
    main()

    with open("avg_reward", "rb") as fp:
        lst = pickle.load(fp)

    plt.plot(lst)
    plt.show()
