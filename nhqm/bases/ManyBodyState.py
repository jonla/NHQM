from itertools import combinations

class FermionState:
    """Represents a many body state"""    
    def __init__(self, states = [], sign = 1):
        self.states = states
        self.sign = sign
    
    def create(self, new_states):
        result_states = self.states
        result_sign = self.sign
        if isinstance(new_states, int):
            new_states = [new_states]
        for new_state in new_states:
            try:
                i = state_index(result_states, new_state)
                result_states.insert(i, new_state)
                result_sign *= (-1)**i            # 
            except IndexError:
                return FermionState(sign = 0)
        return FermionState(result_states, result_sign)
    
    def annihilate(self, kill_states):
        result_states = self.states
        result_sign = self.sign
        if isinstance(kill_states, int):
            kill_states = [kill_states]
        for kill_state in list(kill_states):
            try:
                i = result_states.index(kill_state)
                result_states.pop(i)
                result_sign *= (-1)**i
            except ValueError:
                return FermionState(sign = 0)
        return FermionState(result_states, result_sign)
        
    def __str__(self):
        if self.sign == 0:
            return "0"
        sign = "-" if self.sign < 0 else "" 
        return sign + "| {} >".format(", ".join(str(x) for x in self.states))
    
def state_index(states, new_state):
    for i, state in enumerate(states):
        if new_state == state:
            raise IndexError
        if new_state < state:
            return i
    return len(states)
    
if __name__ == '__main__':
    print FermionState([0,1,3,4]).annihilate(0).create(5)