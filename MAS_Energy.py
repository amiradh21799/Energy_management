import numpy as np
import pandas as pd
from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt

# =========================================================
# Données d'entrée
# =========================================================
# Coûts variables (24 valeurs, une par heure)
C_grid_array = np.array([
    0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
    0.12, 0.12, 0.12, 0.12, 0.12, 0.12,
    0.12, 0.12, 0.12, 0.12, 0.12, 0.12,
    0.12, 0.12, 0.12, 0.12, 0.05, 0.05
])

C_storage_array = np.array([
    0.35, 0.35, 0.35, 0.35, 0.35, 0.35,
    0.40, 0.40, 0.40, 0.40, 0.40, 0.40,
    0.40, 0.40, 0.40, 0.40, 0.40, 0.40,
    0.40, 0.40, 0.40, 0.40, 0.35, 0.35
])

C_pv_array = np.array([
    0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
    0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
    0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
    0.10, 0.10, 0.10, 0.10, 0.10, 0.10
])

# Données de consommation et de production PV
data = pd.DataFrame({
    'timestamp': pd.date_range(start='2019-06-10 00:00:00', periods=24, freq='H'),
    'conso_5RNS': [
        22.973739, 22.624288, 22.099875, 22.966455, 25.806568, 25.916713,
        26.147652, 27.465675, 23.325462, 24.638422, 28.049218, 28.175101,
        24.929978, 25.379126, 24.548666, 28.304361, 27.964538, 28.833112,
        24.124834, 27.444832, 23.589232, 21.945134, 19.556679, 19.946710
    ],
    'conso_HA': [
        46.386624, 45.930994, 45.682043, 45.923036, 47.151663, 48.719195,
        46.429420, 49.086484, 41.955689, 44.469950, 50.096358, 50.877831,
        44.879442, 45.830647, 42.930320, 49.306279, 49.072103, 52.395665,
        42.289834, 48.432067, 53.801493, 46.254925, 40.896865, 43.839718
    ],
    'conso_HEI': [
        82.232439, 80.643039, 80.482353, 79.082003, 86.633257, 89.308227,
        88.459662, 93.198944, 78.139266, 85.725691, 100.251738, 100.892304,
        89.149752, 93.522973, 88.439511, 102.448477, 98.329186, 96.956910,
        75.774048, 91.121469, 99.441998, 90.580749, 81.678680, 77.199761
    ],
    'conso_RIZOMME': [
        20.891538, 20.307705, 19.907527, 20.550549, 18.273802, 17.809438,
        18.241275, 19.887697, 16.579584, 18.568521, 21.602686, 20.054763,
        18.461463, 19.398784, 17.587338, 19.940883, 20.751782, 21.814314,
        17.811283, 21.758343, 23.167278, 20.652838, 17.895999, 19.013811
    ],
    'PV_total': [
        0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 1.20225, 3.81464,
        15.91954, 17.52773, 42.50442, 17.79276, 30.87777, 120.58479,
        100.33294, 138.16286, 120.93127, 130.83308, 80.03456, 40.43099,
        64.65445, 39.25586, 20.79903, 10.34629, 5.00000
    ]
})

# Préférences des différents agents
preferences = {
    'building': [
        {'name': 'Building 1', 'alpha': 0.8, 'beta': 0.1, 'gamma': 0.1},
        {'name': 'Building 2', 'alpha': 0.8, 'beta': 0.1, 'gamma': 0.1},
        {'name': 'Building 3', 'alpha': 0.8, 'beta': 0.1, 'gamma': 0.1},
        {'name': 'Building 4', 'alpha': 0.8, 'beta': 0.1, 'gamma': 0.1}
    ],
    'EV': [
        {'name': 'EV 1', 'alpha': 0.8, 'beta': 0.1, 'gamma': 0.1},
        {'name': 'EV 2', 'alpha': 0.8, 'beta': 0.1, 'gamma': 0.1},
        {'name': 'EV 3', 'alpha': 0.8, 'beta': 0.1, 'gamma': 0.1},
        {'name': 'EV 4', 'alpha': 0.8, 'beta': 0.1, 'gamma': 0.1},
    ],
    'storage': [
        {'name': 'Storage', 'alpha': 0.8, 'beta': 0.2, 'gamma': 0.0}
    ],
    'PV': [
        {'name': 'PV', 'alpha': 0.8, 'beta': 0.1, 'gamma': 0.1},
    ]
}

# =========================================================
# Définition des Agents
# =========================================================

class BuildingAgent(Agent):
    def __init__(self, unique_id, model, demand_profile, preferences, name):
        super().__init__(unique_id, model)
        self.demand_profile = demand_profile
        self.preferences = preferences
        self.name = name
        
        self.consumed_from_pv = 0
        self.consumed_from_storage = 0
        self.consumed_from_grid = 0
        
        self.cost_pv = 0
        self.cost_storage = 0
        self.cost_grid = 0
        self.total_cost = 0
        
        self.pv_used = 0
        self.total_demand = sum(demand_profile)
        self.total_consumed = 0
        self.interactions = {}

    def interact(self, other_agent):
        self.interactions[other_agent.name] += 1
        other_agent.interactions[self.name] += 1

    def step(self):
        t = self.model.schedule.steps
        
        # Récupération des coûts variables pour l'heure t
        cost_grid_current = self.model.C_grid[t]
        cost_pv_current = self.model.C_pv[t]
        cost_storage_current = self.model.C_storage[t]

        demand = self.demand_profile[t]

        # Contribution PV
        pv_contribution = min(self.model.pv_agent.available_power * self.preferences['beta'], demand)
        self.consumed_from_pv = pv_contribution
        demand -= pv_contribution
        self.model.pv_agent.allocate_power(pv_contribution)

        # Contribution stockage
        storage_contribution = 0
        grid_contribution = 0

        if demand > 0 and self.model.storage_agent.soc > self.model.storage_agent.soc_min:
            storage_contribution = min(self.model.storage_agent.available_discharge_power * self.preferences['gamma'],
                                       demand)
            self.consumed_from_storage = storage_contribution
            demand -= storage_contribution
            self.model.storage_agent.discharge(storage_contribution)

        # Contribution réseau
        if demand > 0:
            grid_contribution = demand * self.preferences['alpha']
            self.consumed_from_grid = grid_contribution
            self.model.grid_agent.consume_power(grid_contribution)

        # Enregistrement des interactions
        self.interact(self.model.pv_agent)
        self.interact(self.model.storage_agent)
        self.interact(self.model.grid_agent)

        # Correction d'éventuels déséquilibres
        total_supply = self.consumed_from_pv + self.consumed_from_storage + self.consumed_from_grid
        total_demand = self.demand_profile[t]

        while not np.isclose(total_supply, total_demand, atol=1e-2):
            imbalance = total_supply - total_demand
            if imbalance > 0:  # Surproduction
                if self.consumed_from_storage > 0:
                    reduction = min(self.consumed_from_storage, imbalance)
                    self.consumed_from_storage -= reduction
                    imbalance -= reduction
                if imbalance > 0 and self.consumed_from_grid > 0:
                    self.consumed_from_grid = max(0, self.consumed_from_grid - imbalance)
            else:  # Sous-production
                additional_demand = -imbalance
                # Vérifie la limite de puissance max du grid
                if self.model.grid_agent.consumed_power + additional_demand <= self.model.grid_agent.max_power:
                    self.consumed_from_grid += additional_demand
                else:
                    max_grid_addition = self.model.grid_agent.max_power - self.consumed_from_grid
                    self.consumed_from_grid += max_grid_addition
                    self.consumed_from_storage += (additional_demand - max_grid_addition)

            # Recalcule supply et demand
            total_supply = self.consumed_from_pv + self.consumed_from_storage + self.consumed_from_grid
            total_demand = self.demand_profile[t]

        # S'assure que rien n'est négatif
        self.consumed_from_storage = max(0, self.consumed_from_storage)
        self.consumed_from_grid = max(0, self.consumed_from_grid)

        # Mise à jour consommation totale
        self.total_consumed += total_supply

        # Calcul des coûts (en fonction de l'heure)
        self.cost_pv = self.consumed_from_pv * cost_pv_current
        self.cost_storage = self.consumed_from_storage * cost_storage_current
        self.cost_grid = self.consumed_from_grid * cost_grid_current
        self.total_cost += self.cost_pv + self.cost_storage + self.cost_grid

        # PV utilisée
        self.pv_used += self.consumed_from_pv

        # Debug
        print(
            f"Hour {t}: {self.name} consumed {self.demand_profile[t]:.2f} kWh, "
            f"cost: ${-self.total_cost:.2f}, sources: {{"
            f"'PV': {self.consumed_from_pv:.2f} kWh (${self.cost_pv:.2f}), "
            f"'Storage': {self.consumed_from_storage:.2f} kWh (${self.cost_storage:.2f}), "
            f"'Grid': {self.consumed_from_grid:.2f} kWh (${self.cost_grid:.2f})}}"
        )

    def finalize(self):
        # Confort = ratio énergie reçue / énergie demandée
        self.comfort = self.total_consumed / self.total_demand if self.total_demand > 0 else 0


class EVAgent(Agent):
    def __init__(self, unique_id, model, soc, soc_required, max_power, eta, preferences, arrival, departure, name):
        super().__init__(unique_id, model)
        self.soc = soc
        self.soc_required = soc_required
        self.soc_max = 1.0
        
        self.max_power = max_power
        self.eta = eta
        self.preferences = preferences
        
        self.arrival = arrival
        self.departure = departure
        self.name = name
        
        self.consumed_from_pv = 0
        self.consumed_from_storage = 0
        self.consumed_from_grid = 0
        
        self.cost_pv = 0
        self.cost_storage = 0
        self.cost_grid = 0
        self.total_cost = 0
        
        self.pv_used = 0
        self.interactions = {}

    def interact(self, other_agent):
        self.interactions[other_agent.name] += 1
        other_agent.interactions[self.name] += 1

    def step(self):
        t = self.model.schedule.steps
        
        # Coûts variables pour l'heure t
        cost_grid_current = self.model.C_grid[t]
        cost_pv_current = self.model.C_pv[t]
        cost_storage_current = self.model.C_storage[t]
        
        # L'EV ne charge que pendant son intervalle d'arrivée/départ
        if self.arrival <= t < self.departure and self.soc < self.soc_required:
            # Besoin de charge
            demand = (self.soc_required - self.soc) * (self.max_power / self.eta)

            # PV
            pv_contribution = min(self.model.pv_agent.available_power * self.preferences['beta'], demand)
            self.soc += pv_contribution * self.eta
            self.soc = min(self.soc, self.soc_max)
            self.model.pv_agent.allocate_power(pv_contribution)
            demand -= pv_contribution

            # Stockage
            storage_contribution = 0
            if demand > 0 and self.model.storage_agent.soc > self.model.storage_agent.soc_min:
                storage_contribution = min(
                    self.model.storage_agent.available_discharge_power * self.preferences['gamma'],
                    demand
                )
                self.soc += storage_contribution * self.eta
                self.soc = min(self.soc, self.soc_max)
                self.model.storage_agent.discharge(storage_contribution)
                demand -= storage_contribution

            # Réseau
            grid_contribution = 0
            if demand > 0:
                grid_contribution = demand * self.preferences['alpha']
                self.soc += grid_contribution * self.eta
                self.soc = min(self.soc, self.soc_max)
                self.model.grid_agent.consume_power(grid_contribution)

            # Enregistrement des consommations
            self.consumed_from_pv = pv_contribution
            self.consumed_from_storage = storage_contribution
            self.consumed_from_grid = grid_contribution

            # Interactions
            self.interact(self.model.pv_agent)
            self.interact(self.model.storage_agent)
            self.interact(self.model.grid_agent)

            # Coûts
            self.cost_pv = self.consumed_from_pv * cost_pv_current
            self.cost_storage = self.consumed_from_storage * cost_storage_current
            self.cost_grid = self.consumed_from_grid * cost_grid_current
            self.total_cost += self.cost_pv + self.cost_storage + self.cost_grid

            # PV utilisée
            self.pv_used += self.consumed_from_pv

            print(
                f"Hour {t}: {self.name} charged, sources: "
                f"{{'PV': {pv_contribution:.2f} kWh (${self.cost_pv:.2f}), "
                f"'Storage': {storage_contribution:.2f} kWh (${self.cost_storage:.2f}), "
                f"'Grid': {grid_contribution:.2f} kWh (${self.cost_grid:.2f})}}"
            )

    def finalize(self):
        self.soc = min(self.soc, self.soc_max)
        self.comfort = self.soc / self.soc_max if self.soc_max > 0 else 0
        print(
            f"{self.name} final cost: ${-self.total_cost:.2f}, "
            f"comfort: {self.comfort:.2f}, PV used: {self.pv_used:.2f} kWh"
        )


class StorageAgent(Agent):
    def __init__(self, unique_id, model, soc, max_soc, min_soc, charge_power, discharge_power, efficiency, name):
        super().__init__(unique_id, model)
        self.soc = soc
        self.max_soc = max_soc
        self.soc_min = min_soc
        
        self.charge_power = charge_power
        self.discharge_power = discharge_power
        self.efficiency = efficiency
        
        self.name = name
        
        self.consumed_from_pv = 0
        self.consumed_from_grid = 0
        
        self.cost_pv = 0
        self.cost_grid = 0
        self.revenue_storage = 0
        
        self.total_cost = 0
        self.pv_used = 0
        self.interactions = {}

    def interact(self, other_agent):
        self.interactions[other_agent.name] += 1
        other_agent.interactions[self.name] += 1

    @property
    def available_discharge_power(self):
        # Puisance disponible à la décharge
        return min(self.discharge_power, (self.soc - self.soc_min) * self.efficiency)

    def discharge(self, power):
        self.soc = max(self.soc_min, self.soc - power / self.efficiency)
        # Le "gain" du stockage est calculé avec le même coût unitaire
        # Mais comme c'est un coût négatif pour l'agent, on l'additionne en "revenue"
        # (ça dépend de la convention adoptée)
        current_cost_storage = self.model.C_storage[self.model.schedule.steps]
        self.revenue_storage += power * current_cost_storage

    def step(self):
        t = self.model.schedule.steps
        # Coûts variables
        cost_grid_current = self.model.C_grid[t]
        cost_pv_current = self.model.C_pv[t]
        
        # Si on peut encore charger
        if self.soc < self.max_soc:
            charge_power = min(self.charge_power, (self.max_soc - self.soc) / self.efficiency)
            
            # Charge d'abord depuis le PV (s'il y a de la dispo)
            pv_contribution = min(self.model.pv_agent.available_power, charge_power)
            grid_contribution = charge_power - pv_contribution
            
            # Mise à jour de la SOC
            self.soc += pv_contribution * self.efficiency
            self.soc += grid_contribution * self.efficiency
            
            # PV allouée
            self.model.pv_agent.allocate_power(pv_contribution)
            
            # Enregistrement des consommations
            self.consumed_from_pv = pv_contribution
            self.consumed_from_grid = grid_contribution

            # Interactions
            self.interact(self.model.pv_agent)
            self.interact(self.model.grid_agent)

            # Coûts
            self.cost_pv = self.consumed_from_pv * cost_pv_current
            self.cost_grid = self.consumed_from_grid * cost_grid_current
            self.total_cost += self.cost_pv + self.cost_grid

            # PV utilisée
            self.pv_used += self.consumed_from_pv

            print(
                f"Hour {t}: {self.name} charged from PV with {pv_contribution:.2f} kWh, "
                f"cost: ${self.cost_pv:.2f}. SOC is now {self.soc:.2f}"
            )

    def finalize(self):
        # Coût net = coûts d'achat (total_cost) - revenu de décharge
        self.total_cost = self.total_cost - self.revenue_storage
        self.comfort = self.soc / self.max_soc if self.max_soc > 0 else 0


class PVAgent(Agent):
    def __init__(self, unique_id, model, power_profile, name):
        super().__init__(unique_id, model)
        self.power_profile = power_profile
        self.available_power = 0
        
        self.revenue = 0
        self.energy_sold = 0
        
        self.name = name
        self.interactions = {}

    def interact(self, other_agent):
        self.interactions[other_agent.name] += 1
        other_agent.interactions[self.name] += 1

    def allocate_power(self, power):
        # Coût (ou revenu pour le PV) : on utilise le coût PV à l’heure courante
        current_cost_pv = self.model.C_pv[self.model.schedule.steps]
        self.available_power = max(0, self.available_power - power)
        self.revenue += power * current_cost_pv
        self.energy_sold += power

    def step(self):
        t = self.model.schedule.steps
        self.available_power = self.power_profile[t]
        print(f"Hour {t}: {self.name} generated {self.available_power:.2f} kWh")

    def finalize(self):
        pass


class GridAgent(Agent):
    def __init__(self, unique_id, model, max_power, name):
        super().__init__(unique_id, model)
        self.max_power = max_power
        self.consumed_power = 0
        
        self.total_cost = 0
        self.energy_supplied = 0
        
        self.name = name
        self.interactions = {}

    def interact(self, other_agent):
        self.interactions[other_agent.name] += 1
        other_agent.interactions[self.name] += 1

    def consume_power(self, power):
        # Vérifie limite de puissance souscrite
        if self.consumed_power + power > self.max_power:
            raise ValueError("Grid power exceeds subscribed limit.")
        
        cost_grid_current = self.model.C_grid[self.model.schedule.steps]
        self.consumed_power += power
        self.total_cost += power * cost_grid_current
        self.energy_supplied += power

    def step(self):
        # Remise à zéro de la puissance consommée chaque heure
        self.consumed_power = 0

    def finalize(self):
        pass


# =========================================================
# Définition du Modèle
# =========================================================
class EnergyManagementModel(Model):
    def __init__(
        self, 
        data, 
        preferences,
        soc_ev, 
        soc_storage, 
        soc_min_storage, 
        soc_max_storage
    ):
        super().__init__()
        self.schedule = BaseScheduler(self)
        
        # On suppose 24 pas de temps (24 heures)
        self.delta_t = 1
        
        # =====================================================
        # Stockage des tableaux de coûts dans le modèle
        # =====================================================
        self.C_grid = C_grid_array
        self.C_pv = C_pv_array
        self.C_storage = C_storage_array

        # =====================================================
        # Initialisation des agents
        # =====================================================
        # 1) Agent PV
        self.pv_agent = PVAgent(
            unique_id=0,
            model=self,
            power_profile=data['PV_total'].to_numpy(),
            name="PV"
        )
        self.schedule.add(self.pv_agent)

        # 2) Agent Stockage
        self.storage_agent = StorageAgent(
            unique_id=1,
            model=self,
            soc=soc_storage,
            max_soc=soc_max_storage,
            min_soc=soc_min_storage,
            charge_power=40,         # kW
            discharge_power=80,      # kW
            efficiency=0.98,
            name="Storage"
        )
        self.schedule.add(self.storage_agent)

        # 3) Agent Réseau
        self.grid_agent = GridAgent(
            unique_id=2,
            model=self,
            max_power=580,
            name="Grid"
        )
        self.schedule.add(self.grid_agent)

        # 4) Agents Bâtiments
        self.building_agents = []
        for i in range(4):
            building_agent = BuildingAgent(
                unique_id=i + 3,
                model=self,
                demand_profile=data.iloc[:, i + 1].to_numpy(),
                preferences=preferences['building'][i],
                name=f"Building {i + 1}"
            )
            self.schedule.add(building_agent)
            self.building_agents.append(building_agent)

        # 5) Agents Véhicules Électriques (EV)
        # Exemple de caractéristiques et créneaux d'arrivée/départ
        self.ev_agents = []
        ev_characteristics = [
            {'soc': 0.2, 'soc_required': 0.6, 'max_power': 22, 'eta': 0.8, 'arrival': 9.25, 'departure': 12.5, 'name': 'EV 1'},
            {'soc': 0.45, 'soc_required': 0.8, 'max_power': 22, 'eta': 0.8, 'arrival': 10.5, 'departure': 19.0, 'name': 'EV 2'},
            {'soc': 0.4, 'soc_required': 0.8, 'max_power': 7.2, 'eta': 0.8, 'arrival': 11.0, 'departure': 16.5, 'name': 'EV 3'},
            {'soc': 0.25, 'soc_required': 0.85, 'max_power': 22, 'eta': 0.8, 'arrival': 10.0, 'departure': 18.0, 'name': 'EV 4'},
        ]
        
        for i, ev in enumerate(ev_characteristics):
            ev_agent = EVAgent(
                unique_id=i + 7,
                model=self,
                soc=ev['soc'],
                soc_required=ev['soc_required'],
                max_power=ev['max_power'],
                eta=ev['eta'],
                preferences=preferences['EV'][i],
                arrival=ev['arrival'],
                departure=ev['departure'],
                name=ev['name']
            )
            self.schedule.add(ev_agent)
            self.ev_agents.append(ev_agent)

        # Création d'un dictionnaire d'interactions pour chaque agent
        for agent in self.schedule.agents:
            agent.interactions = {
                other_agent.name: 0 for other_agent in self.schedule.agents
            }

        # DataCollector pour suivre les interactions
        self.datacollector = DataCollector(
            model_reporters={
                "Total Interactions": self.get_total_interactions,
            },
            agent_reporters={
                "Interactions": lambda agent: sum(agent.interactions.values())
            }
        )

    def get_total_interactions(self):
        return sum(sum(agent.interactions.values()) for agent in self.schedule.agents)

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


# =========================================================
# Exécution de l'exemple
# =========================================================

# Conditions initiales pour la batterie stationnaire et les VE
SOC_EV = np.array([0.2, 0.45, 0.4, 0.25])
SOC_storage = 0.35
SOC_min_storage = 0.1
SOC_max_storage = 1.0

model = EnergyManagementModel(
    data=data,
    preferences=preferences,
    soc_ev=SOC_EV,
    soc_storage=SOC_storage,
    soc_min_storage=SOC_min_storage,
    soc_max_storage=SOC_max_storage
)

# Boucle sur 24 heures
for i in range(24):
    print(f"====== Hour {i} ======")
    model.step()

# Finalisation et collecte des données
for agent in model.schedule.agents:
    if isinstance(agent, (BuildingAgent, EVAgent, StorageAgent, PVAgent)):
        agent.finalize()

# Récupération des résultats dans un DataFrame
results = []
for agent in model.schedule.agents:
    if isinstance(agent, BuildingAgent):
        results.append({
            'User Name': agent.preferences['name'],
            'Cost ($)': -agent.total_cost,  # On met le coût en négatif pour rester cohérent avec le signe
            'Comfort': agent.comfort,
            'PV Used (kWh)': agent.pv_used
        })
    elif isinstance(agent, EVAgent):
        results.append({
            'User Name': agent.preferences['name'],
            'Cost ($)': -agent.total_cost,
            'Comfort': agent.comfort,
            'PV Used (kWh)': agent.pv_used
        })
    elif isinstance(agent, StorageAgent):
        results.append({
            'User Name': 'Storage',
            'Cost ($)': agent.total_cost,
            'Comfort': agent.comfort,
            'PV Used (kWh)': agent.pv_used
        })
    elif isinstance(agent, PVAgent):
        results.append({
            'User Name': 'PV',
            'Cost ($)': agent.revenue,  # Revenue PV
            'Comfort': None,
            'PV Used (kWh)': agent.energy_sold
        })

results_df = pd.DataFrame(results)
average_comfort = results_df['Comfort'].dropna().mean()

print("\n===== RESULTS =====")
print(results_df)
print(f"Average Comfort: {average_comfort:.2f}")
total_cost = results_df['Cost ($)'].sum()
print(f"Total Cost: {total_cost:.2f}")

# Exemple de tracé : évolutions des interactions
model_data = model.datacollector.get_model_vars_dataframe()

plt.figure(figsize=(10, 6))
plt.plot(model_data.index, model_data['Total Interactions'], label='Total Interactions')
plt.xlabel('Time Step')
plt.ylabel('Total Interactions')
plt.title('Total Interactions Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Matrice d'interactions (facultatif)
agent_names = [agent.name for agent in model.schedule.agents]
interaction_matrix = np.zeros((len(agent_names), len(agent_names)))

for i, agent in enumerate(model.schedule.agents):
    for j, other_agent in enumerate(model.schedule.agents):
        interaction_matrix[i, j] = agent.interactions[other_agent.name]


hours = range(24)

# Extraction des SOC pour chaque véhicule électrique
ev_socs = {f'EV {i+1}': [] for i in range(4)}

# Dictionnaires pour conserver les profils de puissance pour chaque bâtiment
building_power_profiles = {
    f'Building {i+1}': {'Load': [], 'PV': [], 'Storage': [], 'Grid': []} 
    for i in range(4)
}

# Pour suivre la puissance du stockage et son SOC
storage_power_profile = []
storage_soc = []

# Exécution du modèle une nouvelle fois pour collecter les données horaires afin d'afficher les résultats
model = EnergyManagementModel(data, preferences, SOC_EV, SOC_storage, SOC_min_storage, SOC_max_storage)

for i in hours:
    model.step()
    
    # Récupération de la SOC de chaque EV à chaque heure
    for j, ev_agent in enumerate(model.ev_agents):
        # Si l'heure actuelle est supérieure ou égale à l'heure d'arrivée de l'EV, on lit sa SOC
        if i >= ev_agent.arrival:
            ev_socs[f'EV {j+1}'].append(ev_agent.soc)
        else:
            # Sinon, on répète la dernière valeur connue ou on utilise la SOC initiale si aucune donnée n’est encore disponible
            ev_socs[f'EV {j+1}'].append(
                ev_socs[f'EV {j+1}'][-1] if ev_socs[f'EV {j+1}'] else ev_agent.soc
            )
    
    # Récupération des données de consommation pour chaque bâtiment
    for j, building_agent in enumerate(model.building_agents):
        building_power_profiles[f'Building {j+1}']['Load'].append(building_agent.demand_profile[i])
        building_power_profiles[f'Building {j+1}']['PV'].append(building_agent.consumed_from_pv)
        building_power_profiles[f'Building {j+1}']['Storage'].append(building_agent.consumed_from_storage)
        building_power_profiles[f'Building {j+1}']['Grid'].append(building_agent.consumed_from_grid)
    
    # Calcul de la puissance nette du stockage (charge - décharge)
    charge_power = model.storage_agent.consumed_from_pv + model.storage_agent.consumed_from_grid
    discharge_power = -model.storage_agent.available_discharge_power
    net_power = charge_power + discharge_power
    storage_power_profile.append(net_power)
    
    # Récupération du SOC du stockage
    storage_soc.append(model.storage_agent.soc)

# ==== Affichage des SOC des véhicules électriques sur 24h ====
plt.figure(figsize=(12, 8))
for ev, soc_values in ev_socs.items():
    plt.plot(hours, soc_values, label=f'{ev} SOC', linestyle='-')

plt.title('Évolution de la SOC (State of Charge) des EV sur 24 Heures', fontsize=16)
plt.xlabel('Heure', fontsize=14)
plt.ylabel('État de charge (SOC)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# ==== Affichage des profils de puissance pour chaque bâtiment ====
for building, profiles in building_power_profiles.items():
    plt.figure(figsize=(12, 8))
    plt.plot(hours, profiles['Load'], label='Charge (Load)', linestyle='--', color='grey', linewidth=1.5)
    plt.plot(hours, profiles['PV'], label='Contribution PV', linestyle='-', color='#66c2a5', linewidth=1.5)
    plt.plot(hours, profiles['Storage'], label='Contribution Stockage', linestyle='-', color='#fc8d62', linewidth=1.5)
    plt.plot(hours, profiles['Grid'], label='Contribution Réseau', linestyle='-', color='#8da0cb', linewidth=1.5)

    plt.title(f'Profil de puissance pour {building} sur 24 Heures', fontsize=16)
    plt.xlabel('Heure', fontsize=14)
    plt.ylabel('Puissance (kWh)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

# ==== Affichage du profil de puissance du stockage sur 24h ====
plt.figure(figsize=(12, 8))
plt.plot(hours, storage_power_profile, label='Profil de Puissance du Stockage (kWh)',
         linestyle='-', color='#fc8d62', linewidth=1.5)

plt.title('Profil de Puissance du Stockage sur 24 Heures', fontsize=16)
plt.xlabel('Heure', fontsize=14)
plt.ylabel('Puissance (kWh)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# ==== Affichage de l'état de charge (SOC) du stockage sur 24h ====
plt.figure(figsize=(12, 8))
plt.plot(hours, storage_soc, label='SOC du Stockage', linestyle='-', color='#66c2a5', linewidth=1.5)

plt.title('Évolution de l\'État de Charge (SOC) du Stockage', fontsize=16)
plt.xlabel('Heure', fontsize=14)
plt.ylabel('État de Charge (SOC)', fontsize=14)
plt.ylim(0, 1)  # On suppose que le SOC varie entre 0 et 1
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# ==== Calcul de la consommation totale depuis chaque source pour chaque bâtiment ====
building_source_totals = {}
for building, profiles in building_power_profiles.items():
    total_pv = sum(profiles['PV'])
    total_storage = sum(profiles['Storage'])
    total_grid = sum(profiles['Grid'])

    # Évite les problèmes de NaN en mettant des valeurs non-nulles
    if total_pv + total_storage + total_grid == 0:
        total_pv, total_storage, total_grid = 1, 1, 1

    building_source_totals[building] = {
        'PV': total_pv,
        'Storage': total_storage,
        'Grid': total_grid
    }

    # Préparation des données pour le diagramme circulaire
    labels = ['PV', 'Stockage', 'Réseau']
    sizes = [total_pv, total_storage, total_grid]
    colors = ['#66c2a5', '#fc8d62', '#8da0cb']

    # Affichage du diagramme circulaire
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title(f'Sources de consommation énergétique pour {building}', fontsize=16)
    plt.show()
