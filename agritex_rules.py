class Rule:
    def __init__(self, rule_id, category, description, condition_func, recommendation, priority=1):
        self.id = rule_id
        self.category = category
        self.description = description
        self.condition_func = condition_func
        self.recommendation = recommendation
        self.priority = priority

    def evaluate(self, context):
        try:
            return self.condition_func(context)
        except Exception:
            return False

class AgritexRuleEngine:
    """
    Computable Inference Engine for AGRITEX Guidelines (Zimbabwe).
    Formalizes 50+ agronomic rules into executable logic based on
    SoilGrids and CHIRPS data.
    """
    def __init__(self):
        self.rules = []
        self._init_rules()

    def _init_rules(self):
        self.rules = []
        
        # --- SECTION 1: SOIL ACIDITY & LIMING (Rules 1-5) ---
        self.rules.append(Rule(1, "Acidity", "Severe Acidity", 
            lambda c: c['soil_ph'] < 4.5,
            "CRITICAL: Soil too acidic for most crops. Apply large quantities of Lime (1000kg/ha+) and incorporate organic matter."))
        
        self.rules.append(Rule(2, "Acidity", "Strong Acidity",
            lambda c: 4.5 <= c['soil_ph'] < 5.0,
            "Major Constraint: Apply Dolomitic Lime (500kg/ha) to raise pH and add Magnesium."))
        
        self.rules.append(Rule(3, "Acidity", "Moderate Acidity",
            lambda c: 5.0 <= c['soil_ph'] < 5.5,
            "Liming Recommended: Apply 300kg/ha lime for sensitive crops like Maize and Soybeans."))
        
        self.rules.append(Rule(4, "Acidity", "Optimal pH",
            lambda c: 5.5 <= c['soil_ph'] <= 7.0,
            "Ideal pH range for most Zimbabwean crops. Maintain with maintenance rotation."))
        
        self.rules.append(Rule(5, "Acidity", "Alkaline Soil",
            lambda c: c['soil_ph'] > 7.5,
            "Alkaline conditions detected. Avoid lime. Use acidifying fertilizers (Ammonium Sulphate) if needed."))

        # --- SECTION 2: RAINFALL ZONES / NATURAL REGIONS (Rules 6-10) ---
        # Approximating Zimbabwe's Natural Regions based on mean rainfall
        self.rules.append(Rule(6, "Climate", "Natural Region I/II (High Rainfall)",
            lambda c: c['rainfall_mean'] > 1000,
            "High Potential Zone: Suitable for specialized crops (Tea, Coffee, Macadamia) and intensive Maize."))
        
        self.rules.append(Rule(7, "Climate", "Natural Region II (Good Rainfall)",
            lambda c: 750 <= c['rainfall_mean'] <= 1000,
            "Good Potential: Intensive cropping area. Suitable for Maize, Soybeans, Tobacco, and Cotton."))

        self.rules.append(Rule(8, "Climate", "Natural Region III (Moderate Rainfall)",
            lambda c: 650 <= c['rainfall_mean'] < 750,
            "Semi-Intensive: Rainfall unreliable. Recommend drought-tolerant Maize varieties or supplement with irrigation."))

        self.rules.append(Rule(9, "Climate", "Natural Region IV (Low Rainfall)",
            lambda c: 450 <= c['rainfall_mean'] < 650,
            "Semi-Extensive: High drought risk. Mandatory shift to Small Grains (Sorghum, Pearl Millet) and drought-tolerant legumes."))

        self.rules.append(Rule(10, "Climate", "Natural Region V (Very Low Rainfall)",
            lambda c: c['rainfall_mean'] < 450,
            "Extensive Farming: Not suitable for dryland cropping. Focus on Livestock or irrigation-only production."))

        # --- SECTION 3: MAIZE SUITABILITY (Rules 11-15) ---
        self.rules.append(Rule(11, "Maize", "Premium Maize Zone",
            lambda c: c['rainfall_mean'] > 800 and c['soil_clay'] > 200, # >20% clay
            "Excellent Maize potential. Target yield > 5t/ha. Use long-season varieties (SC700 series)."))

        self.rules.append(Rule(12, "Maize", "Sandy Soil Maize Risk",
            lambda c: c['sand'] > 750 and c['rainfall_mean'] > 600,
            "Sandy soils prone to leaching. Apply Nitrogen in 3 splits. Use nematode resistant varieties."))

        self.rules.append(Rule(13, "Maize", "Marginal Maize Zone",
            lambda c: 500 <= c['rainfall_mean'] < 650,
            "Marginal for Maize. Use ultra-early maturing varieties (SC300/400 series) and water conservation techniques."))

        self.rules.append(Rule(14, "Maize", "Maize Failure Risk",
            lambda c: c['rainfall_mean'] < 500,
            "WARNING: High risk of Maize failure. Do NOT plant Maize. Switch to Sorghum or Millet."))
            
        self.rules.append(Rule(15, "Maize", "Clay Soil Advice",
            lambda c: c['soil_clay'] > 400, # Heavy clay
            "Heavy clay soils: Ensure good drainage to prevent waterlogging for Maize."))

        # --- SECTION 4: SMALL GRAINS (Rules 16-20) ---
        self.rules.append(Rule(16, "Small Grains", "Sorghum Suitability",
            lambda c: 450 <= c['rainfall_mean'] < 700,
            "Highly Recommended: Sorghum is the food security crop of choice here."))

        self.rules.append(Rule(17, "Small Grains", "Pearl Millet Niche",
            lambda c: c['rainfall_mean'] < 500 and c['sand'] > 600,
            "Ideal for Pearl Millet (Mhunga): Thrives in sandy, low rainfall areas where maize fails."))

        self.rules.append(Rule(18, "Small Grains", "Finger Millet",
            lambda c: c['rainfall_mean'] > 650 and c['soil_ph'] < 5.5,
            "Finger Millet (Rapoko) tolerance: Can yield well even in acidic soils where maize struggles."))

        self.rules.append(Rule(19, "Small Grains", "Bird Warning",
            lambda c: True, # General advice
            "Small Grains Alert: Scout for Quelea birds during grain filling stage."))

        self.rules.append(Rule(20, "Small Grains", "Brewing Grade",
            lambda c: c['nitrogen'] > 120 and c['rainfall_mean'] > 600,
            "Conditions suitable for Red Sorghum (Brewing Grade). Contract farming opportunity."))

        # --- SECTION 5: CASH CROPS (Rules 21-25) ---
        self.rules.append(Rule(21, "Cash Crops", "Tobacco Ideal",
            lambda c: c['sand'] > 700 and c['rainfall_mean'] > 700,
            "Tobacco Belt: Sandy loams with good drainage and rainfall. High value potential."))

        self.rules.append(Rule(22, "Cash Crops", "Cotton Clay",
            lambda c: c['soil_clay'] > 300 and c['rainfall_mean'] < 700 and c['rainfall_mean'] > 500,
            "Cotton Suitability: Performs well in heavier soils with moderate rainfall."))

        self.rules.append(Rule(23, "Cash Crops", "Soybean pH sensitivity",
            lambda c: c['soil_ph'] < 5.2,
            "Soybean Risk: Avoid planting legumes until pH is corrected to > 5.5. Rhizobium inoculation will fail."))

        self.rules.append(Rule(24, "Cash Crops", "Groundnut Sandy",
            lambda c: c['sand'] > 650 and c['rainfall_mean'] > 550,
            "Groundnuts: Ideal sandy soils for pegging. Apply Gypsum at flowering (Calcium critical)."))

        self.rules.append(Rule(25, "Cash Crops", "Sunflower Hardiness",
            lambda c: c['rainfall_mean'] < 600 and c['soil_clay'] > 150,
            "Sunflower Option: Deep rooting system handles moisture stress better than Maize in heavy soils."))

        # --- SECTION 6: FERTILITY MANAGEMENT (Rules 26-30) ---
        self.rules.append(Rule(26, "Fertility", "Low Nitrogen",
            lambda c: c.get('nitrogen', 0) < 50,
            "Low Nitrogen Status: Heavy top-dressing required. Split application 50/50 at 4 weeks and 7 weeks."))

        self.rules.append(Rule(27, "Fertility", "High Carbon Sequestration",
            lambda c: c.get('soil_soc', 0) > 60,
            "High Organic Matter: Reduce Nitrogen slightly as mineralization will provide N release."))

        self.rules.append(Rule(28, "Fertility", "Low Organic Matter",
            lambda c: c.get('soil_soc', 0) < 20,
            "Degraded Soil: Structure collapse risk. Must apply manure/compost (5-10t/ha) to restore viability."))

        self.rules.append(Rule(29, "Fertility", "Compaction Risk",
            lambda c: c.get('bulk_density', 0) > 160,
            "Soil Compaction Alert: High bulk density detected. Rip/chisel plough needed. Roots cannot penetrate."))
        
        self.rules.append(Rule(30, "Fertility", "Leaching Risk",
            lambda c: c['sand'] > 850 and c['rainfall_mean'] > 900,
            "Severely Leached Soil: Frequent, small fertilizer applications essential. Use granular slow release if available."))

        # --- SECTION 7: CLIMATE RESILIENCE / CONSERVATION AG (Rules 31-36) ---
        self.rules.append(Rule(31, "Conservation", "Pfumvudza/Basins",
            lambda c: c['rainfall_mean'] < 700,
            "Conservation Agriculture: Adopt 'Pfumvudza' planting basins to harvest water and concentrate nutrients."))
        
        self.rules.append(Rule(32, "Conservation", "Mulching",
            lambda c: c['rainfall_mean'] < 600 or c['rainfall_cv'] > 0.3,
            "Moisture Stress High: Mulching is non-negotiable to prevent evaporation."))

        self.rules.append(Rule(33, "Conservation", "Tie-Ridging",
            lambda c: c['rainfall_mean'] > 800 and c['soil_clay'] > 150,
            "Waterlogging Risk: Use Tie-Ridges (Box Ridges) to retain water in furrows but drain excess during flooding."))

        self.rules.append(Rule(34, "Conservation", "High Variability",
            lambda c: c['rainfall_cv'] > 0.4,
            "Extreme Climate Volatility: Do not rely on a single crop. Intercrop Maize with Cowpeas/Pumpkins for insurance."))
            
        self.rules.append(Rule(35, "Conservation", "Erosion Hazard",
            lambda c: c['rainfall_mean'] > 900 and c['sand'] > 600, # High rain on sand
            "Erosion Warning: Contour ridges must be maintained. Sandy soil + High Rain = Gullies."))

        self.rules.append(Rule(36, "Conservation", "Minimum Tillage",
            lambda c: True,
            "General Rec: Minimize soil disturbance to preserve carbon stock and structure."))

        # --- SECTION 8: IRRIGATION & WATER (Rules 37-40) ---
        self.rules.append(Rule(37, "Water", "Winter Wheat",
            lambda c: False, # Needs explicit irrigation flag, assuming rainfed dataset for now
            "Winter Wheat: Only possible with full irrigation functionality."))

        self.rules.append(Rule(38, "Water", "Supplemental Irrigation",
            lambda c: c['rainfall_mean'] < 600 and c.get('irrigation_potential', False),
            "Investment: Install drip kits for vegetable production to supplement low rainfall."))

        self.rules.append(Rule(39, "Water", "Water Harvesting",
            lambda c: c['rainfall_mean'] < 500,
            "Critical: In-field water harvesting (dead level contours, infiltration pits) required for any yield."))

        self.rules.append(Rule(40, "Water", "Good Rainfed Potential",
            lambda c: c['rainfall_mean'] >= 750,
            "Rainfed Potential: Sufficient for rainfed summer cropping without major intervention."))

        # --- SECTION 9: SPECIALIZED CHECKS (Rules 41-46) ---
        self.rules.append(Rule(41, "Special", "Sodic Soil Risk",
            lambda c: c['soil_ph'] > 8.5,
            "Sodicity Risk: Structure collapse likely. Apply Gypsum. Drainage is critical."))

        self.rules.append(Rule(42, "Special", "Striga Weed Risk",
            lambda c: c['nitrogen'] < 40 and c['sand'] > 600,
            "Striga (Witchweed) Alert: Low N sandy soils favor Striga. Use rotation with 'Trap Crops' like Cotton or Desmodium."))

        self.rules.append(Rule(43, "Special", "Fall Armyworm",
            lambda c: True,
            "Pest Watch: Monitor for Fall Armyworm from emergence. Threshold for action: 5% plants affected."))

        self.rules.append(Rule(44, "Special", "Nematode Risk",
            lambda c: c['sand'] > 800,
            "Nematode Warning: Very sandy soils. Rotate with grass/pasture or Velvet Bean to break cycle."))
            
        self.rules.append(Rule(45, "Special", "High Potential Horticulture",
            lambda c: c['rainfall_mean'] > 900 and 5.5 <= c['soil_ph'] <= 6.5,
            "Horticulture: Ideal conditions for Potatoes, Tomatoes, and Cabbages."))

        self.rules.append(Rule(46, "Special", "Wetland Protection",
            lambda c: c['soil_soc'] > 150, # Peat/Wetland proxy
            "Environmental: High SOC suggests wetland/vlei. Protected area - Do not drain/plough."))

        # --- SECTION 10: LIVESTOCK INTEGRATION (Rules 47-50) ---
        self.rules.append(Rule(47, "Livestock", "Fodder Bank",
            lambda c: c['rainfall_mean'] < 500,
            "Livestock Focus: Prioritize growing fodder crops (Lablab, Velvet Bean) over grain for sale."))

        self.rules.append(Rule(48, "Livestock", "Mixed Farming",
            lambda c: 500 <= c['rainfall_mean'] < 700,
            "Mixed System: Use crop residues for winter feed. Manure return to field is vital."))

        self.rules.append(Rule(49, "Livestock", "Intensive Dairy",
            lambda c: c['rainfall_mean'] > 900,
            "Dairy Potential: High rainfall allows planted pastures (Rhodes Grass, Kikuyu)."))

        self.rules.append(Rule(50, "Livestock", "Carrying Capacity",
            lambda c: c['rainfall_mean'] < 450,
            "Rangeland: Low carrying capacity. 1 LSU per 10-15ha. Do not overstock."))
            
        # Final catch-all
        self.rules.append(Rule(51, "General", "Consultation",
            lambda c: True,
            "Always consult your local AGRITEX Extension Officer for site-specific verification."))

    def evaluate(self, inputs):
        """
        Run forward inference on the rule set.
        inputs: dict of variables (soil_ph, rainfall_mean, sand, soil_clay, etc.)
        """
        results = {
            'recommendations': [],
            'triggered_rules': [],
            'categories': {}
        }
        
        for rule in self.rules:
            if rule.evaluate(inputs):
                # Add to flat lists
                results['recommendations'].append(rule.recommendation)
                results['triggered_rules'].append({
                    'id': rule.id,
                    'category': rule.category,
                    'desc': rule.description,
                    'rec': rule.recommendation
                })
                
                # Group by category
                cat = rule.category
                if cat not in results['categories']:
                    results['categories'][cat] = []
                results['categories'][cat].append(rule.recommendation)
                
        return results

if __name__ == "__main__":
    # Test case representing a random point in Zimbabwe
    test_ctx = {
        'soil_ph': 4.8, 
        'rainfall_mean': 550, 
        'rainfall_cv': 0.35,
        'nitrogen': 45,
        'sand': 780, # 78% sand
        'soil_clay': 120, # 12% clay
        'soil_soc': 15, # Low organic carbon
        'bulk_density': 140
    }
    
    engine = AgritexRuleEngine()
    print(f"Loaded {len(engine.rules)} rules.")
    print("\\nTesting Context:", test_ctx)
    
    res = engine.evaluate(test_ctx)
    print("\\n--- Triggered Recommendations ---")
    for rec in res['recommendations']:
        print(f"- {rec}")
