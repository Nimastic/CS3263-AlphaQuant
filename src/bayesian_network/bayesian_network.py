import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pymc3 as pm


class BayesianNetworkModel:
    def __init__(self):
        self.graph = None
        self.model = None
        self.trace = None
        
    def create_asset_network(self, assets, market_data):
        """Create a Bayesian Network model of asset relationships"""
        # Create directed graph structure
        G = nx.DiGraph()
        
        # Add nodes for each asset
        for asset in assets:
            G.add_node(asset)
        
        # Add market factor nodes
        G.add_node("market_sentiment")
        G.add_node("interest_rates")
        G.add_node("economic_growth")
        
        # Add edges from market factors to assets
        for asset in assets:
            G.add_edge("market_sentiment", asset)
            G.add_edge("interest_rates", asset)
            G.add_edge("economic_growth", asset)
        
        # Add edges between related assets (based on sector)
        sectors = {
            "Technology": ["AAPL", "MSFT", "GOOGL"],
            "Finance": ["JPM", "BAC", "GS"],
            "Energy": ["XOM", "CVX", "COP"]
        }
        
        # Connect assets within same sector
        for sector, stocks in sectors.items():
            sector_assets = [a for a in assets if a in stocks]
            for i in range(len(sector_assets)):
                for j in range(i+1, len(sector_assets)):
                    G.add_edge(sector_assets[i], sector_assets[j])
        
        self.graph = G
        return G
    
    def build_pymc_model(self, market_data, assets):
        """Build PyMC3 model based on network structure"""
        with pm.Model() as model:
            # Market factors as root nodes
            market_sentiment = pm.Normal('market_sentiment', mu=0, sigma=1)
            interest_rates = pm.Normal('interest_rates', mu=0, sigma=1)
            economic_growth = pm.Normal('economic_growth', mu=0, sigma=1)
            
            # Asset returns conditioned on market factors
            asset_returns = {}
            for asset in assets:
                # Regression coefficients (betas)
                beta_sentiment = pm.Normal(f'beta_sentiment_{asset}', mu=0.5, sigma=0.2)
                beta_rates = pm.Normal(f'beta_rates_{asset}', mu=-0.3, sigma=0.2)
                beta_growth = pm.Normal(f'beta_growth_{asset}', mu=0.4, sigma=0.2)
                
                # Residual volatility
                sigma = pm.HalfNormal(f'sigma_{asset}', sigma=0.05)
                
                # Mean return based on factors
                mu = (beta_sentiment * market_sentiment + 
                      beta_rates * interest_rates + 
                      beta_growth * economic_growth)
                
                # Observed returns
                if asset in market_data.columns:
                    asset_returns[asset] = pm.Normal(
                        asset,
                        mu=mu,
                        sigma=sigma,
                        observed=market_data[asset]
                    )
        
        self.model = model
        return model
    
    def fit(self, samples=1000):
        """Fit the model by sampling from posterior"""
        if self.model is None:
            raise ValueError("Model not built. Call build_pymc_model first.")
            
        with self.model:
            self.trace = pm.sample(samples, return_inferencedata=True)
            
        return self.trace
    
    def plot_network(self):
        """Visualize the Bayesian Network structure"""
        if self.graph is None:
            raise ValueError("Graph not created. Call create_asset_network first.")
            
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', 
                node_size=1500, arrowsize=20, font_size=12)
        
        plt.title("Asset Relationship Bayesian Network")
        return plt.gcf() 