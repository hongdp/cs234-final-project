import numpy as np
from .agent import Agent, Actions
from sklearn import linear_model

def is_power2(num):
	'''states if a number is a power of two'''

	return num != 0 and ((num & (num - 1)) == 0)
def is_j(j, i, q):
    return q*i <= j < q*(i+1)

class LassoAgent(Agent):

    def __init__(self, config, vocab):
        self.actions = [Actions.LOW, Actions.MEDIUM, Actions.HIGH]
        self.q = config.lasso_q
        self.h = config.lasso_h
        self.lam1 = config.lasso_lam1
        self.lam2_0 = config.lasso_lam2
        self.lam2 = config.lasso_lam2
        self.t = 0
        self.feature_len = config.feature_len
        self.T_x = [np.array([], dtype=np.float32).reshape((0, self.feature_len))]*3
        self.T_y = [np.array([], dtype=np.float32).reshape((0))]*3
        self.S_x = [np.array([], dtype=np.float32).reshape((0, self.feature_len))]*3
        self.S_y = [np.array([], dtype=np.float32).reshape((0))]*3


    def act(self, feature):
        # if in T
        Kq = len(self.actions) * self.q
        quot = self.t // Kq
        rem = self.t % Kq
        for action in self.actions:
            if is_power2(quot + 1) and is_j(rem, action.value, self.q):
                return action, {'feature': feature, 'forced_sample': True, 'action': action}

        # not in T
        # Prefilter
        T_lasso_predicted_val = []
        feature = np.asarray(feature, dtype=np.float32)
        for action in self.actions:
            lasso_pred = linear_model.Lasso(alpha=self.lam1)
            lasso_pred.fit(self.T_x[action.value], self.T_y[action.value])
            pred_val = lasso_pred.predict(feature.reshape(1,-1))
            T_lasso_predicted_val.append(pred_val)
        T_lasso_max = np.amax(T_lasso_predicted_val)

        available_actions = []
        for action, value in zip(self.actions, T_lasso_predicted_val):
            if value > T_lasso_max - self.h/2:
                available_actions.append(action)

        S_lasso_predicted_max_val = None
        S_lasso_predicted_max_action = None
        for action in available_actions:
            lasso_pred = linear_model.Lasso(alpha=self.lam2)
            lasso_pred.fit(self.S_x[action.value], self.S_y[action.value])
            pred_val = lasso_pred.predict(feature.reshape(1,-1))
            if S_lasso_predicted_max_val == None or pred_val > S_lasso_predicted_max_val:
                S_lasso_predicted_max_val = pred_val
                S_lasso_predicted_max_action = action
        return S_lasso_predicted_max_action, {'feature': feature, 'forced_sample': False, 'action': S_lasso_predicted_max_action}


    def feedback(self, reward, context):
        self.t += 1
        self.lam2 = self.lam2_0 * np.sqrt((np.log(self.t) * np.log(self.feature_len)/self.t))
        if context['forced_sample']:
            self.T_x[context['action'].value] = np.concatenate((self.T_x[context['action'].value], context['feature'].reshape(1, self.feature_len)), axis=0)
            self.T_y[context['action'].value] = np.concatenate((self.T_y[context['action'].value], np.array([reward], dtype=np.float32)))
        self.S_x[context['action'].value] = np.concatenate((self.S_x[context['action'].value], context['feature'].reshape(1, self.feature_len)), axis=0)
        self.S_y[context['action'].value] = np.concatenate((self.S_y[context['action'].value], np.array([reward], dtype=np.float32)))

