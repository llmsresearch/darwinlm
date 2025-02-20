import pytest
from darwinlm.evolution.search import EvolutionarySearch
from darwinlm.database.sparsity_db import SparsityDatabase

@pytest.fixture
def mock_config():
    return {
        "evolution": {
            "num_generations": 2,
            "offspring_size": 2,
            "selection_steps": 2,
            "finetune": {
                "tokens_per_step": [100, 200],
                "selection_tokens": [50, 100]
            }
        }
    }

def test_level_switch_mutation():
    db = SparsityDatabase(num_levels=10)
    search = EvolutionarySearch(db, 2, 2, 2, [100, 200], [50, 100])
    
    parent = [5] * 10  # 10 layers with level 5
    offspring = search.level_switch_mutation(parent)
    
    assert len(offspring) == len(parent)
    assert sum(offspring) == sum(parent)  # Total sparsity preserved
    assert offspring != parent  # Should be different 