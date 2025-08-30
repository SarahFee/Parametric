"""
Integrated Business Continuity Insurance Model with HDX and IATI Data

This model integrates:
1. IATI data for organization profiles and financial patterns
2. HDX data for security and emergency risk parameters (now using ACAPS INFORM Severity Index)
3. Agent-based modeling for simulating insurance dynamics
"""
import random
import logging
import datetime
import os
import sys
from mesa import Model, Agent
from mesa.time import RandomActivation
import numpy as np

# Configure logging
log_directory = 'logs'
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(log_directory, 'integrated_business_model.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger(__name__)
logger.info("--- Integrated business continuity model starting ---")

# Ensure the directory containing these modules is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import API modules with proper error handling - AVOID CIRCULAR IMPORTS
HDX_AVAILABLE = False
try:
    # Import specific functions rather than the whole module
    from hdx_integration import (
        get_all_hdx_parameters, 
        get_security_parameters_from_hdx, 
        get_emergency_parameters_from_hdx,
        fetch_hdx_data
    )
    HDX_AVAILABLE = True
    logger.info("Successfully imported HDX integration module")
except ImportError as e:
    logger.warning(f"Could not import HDX integration module: {e}. HDX data will not be used.")
    
    # Define fallback functions
    def get_all_hdx_parameters(): return None
    def get_security_parameters_from_hdx(): return None
    def get_emergency_parameters_from_hdx(): return None
    def fetch_hdx_data(api_key, country_code="SDN", show_spinner=False): return None

IATI_AVAILABLE = False
try:
    from iati_api import (
        get_iati_parameters_for_model, 
        get_organization_definitions_from_iati,
        get_donor_definitions_from_iati
    )
    IATI_AVAILABLE = True
    logger.info("Successfully imported IATI API module")
except ImportError as e:
    logger.warning(f"Could not import IATI API module: {e}. IATI data will not be used.")
    
    # Define fallback functions
    def get_iati_parameters_for_model():
        logger.debug("Using default fallback for IATI parameters.")
        return {
            'organizations': {
                "NGO": {'initial_capital': 4500000, 'total_budget': 8000000, 'operational_rate': 0.010,
                       'premium_discount': 0.9, 'claim_trigger': 0.4, 'security_risk_modifier': 1.0,
                       'emergency_probability_modifier': 1.2, 'emergency_impact': 0.08},
                "UN Agency": {'initial_capital': 8000000, 'total_budget': 15000000, 'operational_rate': 0.008,
                             'premium_discount': 1.0, 'claim_trigger': 0.55, 'security_risk_modifier': 0.5,
                             'emergency_probability_modifier': 0.7, 'emergency_impact': 0.15},
                "Hybrid": {'initial_capital': 6000000, 'total_budget': 10000000, 'operational_rate': 0.01,
                          'premium_discount': 0.95, 'claim_trigger': 0.5, 'security_risk_modifier': 0.8,
                          'emergency_probability_modifier': 1.0, 'emergency_impact': 0.10}
            },
            'risks': {'emergency_probability': 0.035, 'security_risk_factor': 0.12,
                     'waiting_period': 2, 'payout_cap_multiple': 3.0},
            'sim_duration': 12
        }
    
    def get_organization_definitions_from_iati():
        logger.debug("Using default fallback for IATI organization definitions.")
        return [
            {"name": "World-Vision", "org_type": "NGO", "initial_capital": 4500000, "total_budget": 8000000},
            {"name": "UNHCR", "org_type": "UN Agency", "initial_capital": 8000000, "total_budget": 15000000},
            {"name": "IOM", "org_type": "Hybrid", "initial_capital": 6000000, "total_budget": 10000000}
        ]
        
    def get_donor_definitions_from_iati():
        logger.debug("Using default fallback for IATI donor definitions.")
        return [
            {"name": "USAID", "donor_type": "11", "avg_donation": 1500000, "donation_frequency": 3,
             "target_preferences": {"NGO": 0.4, "UN Agency": 0.4, "Hybrid": 0.2}},
            {"name": "ECHO", "donor_type": "40", "avg_donation": 1200000, "donation_frequency": 2.5,
             "target_preferences": {"NGO": 0.3, "UN Agency": 0.5, "Hybrid": 0.2}},
            {"name": "Private Foundation", "donor_type": "60", "avg_donation": 500000, "donation_frequency": 4,
             "target_preferences": {"NGO": 0.7, "UN Agency": 0.1, "Hybrid": 0.2}}
        ]
    # This should be at the module level, not inside another function
def fetch_hdx_data(api_key, country_code="SDN", show_spinner=False):
    """
    Fetch HDX data using the provided API key.
    Returns the fetched data or None if fetching fails.
    """
    if not HDX_AVAILABLE or not api_key:
        logger.warning("HDX integration not available or no API key provided.")
        return None
        
    try:
        # Validate API key
        if not isinstance(api_key, str) or len(api_key.strip()) < 5:
            logger.error("Invalid API key format.")
            return None

        # Try different function names
        try:
            # Attempt to import and use fetch functions dynamically
            fetch_fn = None
            fn_name = None

            # Check for various possible fetch function names
            possible_fn_names = [
                'fetch_hdx_data', 
                'fetch_hapi_data', 
                'get_hdx_parameters', 
                'retrieve_hdx_data'
            ]

            for potential_name in possible_fn_names:
                if hasattr(hdx_integration, potential_name):
                    fetch_fn = getattr(hdx_integration, potential_name)
                    fn_name = potential_name
                    break

            if not fetch_fn:
                logger.error("No suitable HDX data fetch function found in hdx_integration module.")
                return None

            # Log detailed fetch attempt
            logger.info(f"Attempting to fetch HDX data:")
            logger.info(f"  Function: {fn_name}")
            logger.info(f"  API Key: {api_key[:4]}...")
            logger.info(f"  Country Code: {country_code}")

            # Flexible parameter passing
            try:
                # Try with multiple parameter combinations
                param_combinations = [
                    {'api_key': api_key, 'country_code': country_code},
                    {'api_key': api_key, 'country': country_code},
                    {'key': api_key, 'country_code': country_code},
                    {'api_key': api_key},
                    {}  # Last resort: no parameters
                ]

                for params in param_combinations:
                    try:
                        hdx_data = fetch_fn(**params)
                        if hdx_data:
                            break
                    except TypeError:
                        # If parameter combination doesn't match function signature, continue
                        continue

                if not hdx_data:
                    logger.warning("Could not fetch HDX data with any parameter combination.")
                    return None

                logger.info("Successfully fetched HDX data")
                
                # Extensive logging of fetched data structure
                logger.info("Fetched HDX Data Structure:")
                logger.info(f"  Keys: {list(hdx_data.keys())}")
                
                # Log data sources
                sec_source = hdx_data.get("security_parameters", {}).get("data_source", "Unknown")
                emg_source = hdx_data.get("emergency_parameters", {}).get("data_source", "Unknown")
                logger.info(f"  Data sources:")
                logger.info(f"    Security: {sec_source}")
                logger.info(f"    Emergency: {emg_source}")

                return hdx_data

            except Exception as call_e:
                logger.error(f"Error calling HDX data fetch function {fn_name}: {call_e}", exc_info=True)
                return None

        except Exception as inner_e:
            logger.error(f"Unexpected error in HDX data fetch: {inner_e}", exc_info=True)
            return None
            
    except Exception as e:
        logger.error(f"Critical error fetching HDX data: {e}", exc_info=True)
        return None

# Define organization types for clarity
ORG_TYPES = ["NGO", "UN Agency", "Hybrid"]

class DonorAgent(Agent):
    """
    Agent representing a donor that provides funding to humanitarian organizations.
    Uses IATI data for donor characteristics when available.
    """
    def __init__(self, unique_id, model, name, donor_type="10", avg_donation=1000000,
                 donation_frequency=3, target_preferences=None):
        super().__init__(unique_id, model)
        self.name = name
        self.donor_type = donor_type  # IATI donor type code
        self.avg_donation = avg_donation  # Average donation amount
        self.donation_frequency = donation_frequency  # Average months between donations

        # Target preferences (probability of donating to each organization type)
        default_prefs = {"NGO": 0.3, "UN Agency": 0.3, "Hybrid": 0.4}
        self.target_preferences = target_preferences if isinstance(target_preferences, dict) else default_prefs
        
        # Normalize preferences if provided
        if isinstance(target_preferences, dict):
            total_pref = sum(target_preferences.values())
            if total_pref > 0 and abs(total_pref - 1.0) > 1e-6:  # Check if normalization needed
                logger.warning(f"Donor {name} preferences sum to {total_pref}, normalizing.")
                self.target_preferences = {k: v / total_pref for k, v in target_preferences.items()}
            elif total_pref == 0:
                logger.warning(f"Donor {name} preferences sum to 0, using defaults.")
                self.target_preferences = default_prefs

        # Donation counter for tracking when to donate
        self.steps_since_last_donation = random.randint(0, int(self.donation_frequency))  # Start at random point in cycle

        # Initialize donation history
        self.donation_history = []

        logger.info(f"Created Donor Agent: {name}, Type: {donor_type}")
        logger.info(f"  Average Donation: ${self.avg_donation:,.0f}")
        logger.info(f"  Donation Frequency: Every {self.donation_frequency:.1f} months")
        logger.info(f"  Target Preferences: {self.target_preferences}")

    def step(self):
        """Perform donor's actions for the current step - potentially make a donation."""
        # Increment counter
        self.steps_since_last_donation += 1

        # Improved donation probability formula
        base_prob = self.steps_since_last_donation / self.donation_frequency
        
        # Increase donation probability when orgs are low on funds
        orgs_in_need = 0
        total_orgs = 0
        
        for org in self.model.orgs:
            total_orgs += 1
            if org.balance < org.total_budget * 0.4:  # Organization is low on funds
                orgs_in_need += 1
        
        need_factor = 1.0 + (orgs_in_need / max(1, total_orgs)) * 0.5  # Scale factor based on need
        prob_donate = min(0.9, base_prob * need_factor)  # Cap at 90% probability
        
        # More regular donations
        force_donation = self.steps_since_last_donation >= self.donation_frequency * 1.2
        
        if random.random() < prob_donate or force_donation:
            try:
                # Determine donation amount with some variability
                # Larger base amounts to improve sustainability
                donation_amount = max(1000, self.avg_donation * random.uniform(0.8, 1.4))  # Ensure min amount

                # Group organizations by type (ensure model.orgs exists)
                orgs_by_type = {}
                if not hasattr(self.model, 'orgs') or not self.model.orgs:
                    logger.warning(f"Donor {self.name}: No organizations available in model to donate to.")
                    return  # Cannot donate if no orgs

                for org in self.model.orgs:
                    if org.org_type not in orgs_by_type:
                        orgs_by_type[org.org_type] = []
                    orgs_by_type[org.org_type].append(org)

                # Select target organization type based on preferences
                available_types = [t for t in self.target_preferences if t in orgs_by_type and orgs_by_type[t]]
                if not available_types:
                    logger.warning(f"Donor {self.name}: No organizations match preference types: {list(self.target_preferences.keys())}. Donating randomly.")
                    # Fallback: select randomly from any available org
                    selected_org = random.choice(self.model.orgs)
                else:
                    # Filter preferences to available types and re-normalize weights
                    filtered_prefs = {t: self.target_preferences[t] for t in available_types}
                    total_weight = sum(filtered_prefs.values())
                    if total_weight == 0:  # Handle case where available types have 0 preference
                        logger.warning(f"Donor {self.name}: Available types {available_types} have 0 preference weight. Donating randomly among them.")
                        selected_type = random.choice(available_types)
                    else:
                        weights = [filtered_prefs[t] / total_weight for t in available_types]
                        selected_type = random.choices(available_types, weights=weights, k=1)[0]

                    # Select a random org of the chosen type
                    selected_org = random.choice(orgs_by_type[selected_type])

                # Increase donation amounts for orgs in financial trouble
                if hasattr(selected_org, 'balance') and hasattr(selected_org, 'total_budget'):
                    if selected_org.balance < selected_org.total_budget * 0.3:
                        emergency_bonus = 1.7  # 70% bonus for orgs in financial trouble
                        original_amount = donation_amount
                        donation_amount *= emergency_bonus
                        log_msg = f"Donor {self.name} increased donation to struggling {selected_org.name} from ${original_amount:,.0f} to ${donation_amount:,.0f} (+{(emergency_bonus-1)*100:.0f}%)."
                        logger.info(log_msg)
                        self.model.log_event(log_msg)

                # Make the donation
                selected_org.receive_funds(donation_amount)

                # Record the donation
                self.donation_history.append({
                    "step": self.model.current_time,
                    "recipient": selected_org.name,
                    "amount": donation_amount
                })

                log_msg = f"Donor {self.name} donated ${donation_amount:,.0f} to {selected_org.name} ({selected_org.org_type})."
                logger.info(log_msg)
                self.model.log_event(log_msg)

                # Reset counter
                self.steps_since_last_donation = 0

            except Exception as e:
                logger.error(f"Error in donor step for {self.name}: {e}", exc_info=True)

    def get_donation_statistics(self):
        """Get statistics about this donor's donation patterns"""
        if not self.donation_history:
            return {
                "total_donated": 0,
                "num_donations": 0,
                "avg_donation": 0,
                "recipient_breakdown": {}
            }

        total_donated = sum(d["amount"] for d in self.donation_history)
        num_donations = len(self.donation_history)

        # Calculate recipient breakdown
        recipient_breakdown = {}
        for donation in self.donation_history:
            recipient = donation["recipient"]
            recipient_breakdown[recipient] = recipient_breakdown.get(recipient, 0) + donation["amount"]

        return {
            "total_donated": total_donated,
            "num_donations": num_donations,
            "avg_donation": total_donated / num_donations if num_donations > 0 else 0,
            "recipient_breakdown": recipient_breakdown
        }

class OrgAgent(Agent):
    """
    Agent representing a humanitarian organization.
    Uses IATI data for financial profiles and HDX data for risk factors when available.
    """
    def __init__(self, unique_id, model, name, org_type, initial_capital, total_budget,
                 operational_rate=None, premium_discount=None, claim_trigger=None,
                 security_risk_modifier=None, emergency_probability_modifier=None,
                 emergency_impact=None):
        super().__init__(unique_id, model)

        # Logging initialization details
        logger.info(f"Initializing Organization Agent: {name} ({org_type})")
        logger.info(f"  Initial Config:")
        logger.info(f"    Initial Capital: ${initial_capital:,.2f}")
        logger.info(f"    Total Budget: ${total_budget:,.2f}")

        self.name = name
        self.org_type = org_type
        self.balance = initial_capital
        self.initial_capital = initial_capital  # Store original initial capital
        self.total_budget = total_budget

        # --- Parameter Selection Logic ---
        # Priority: User > HDX > IATI > Default
        # Start with None or user-provided values
        selected_op_rate = operational_rate
        selected_prem_disc = premium_discount
        selected_claim_trig = claim_trigger
        selected_sec_mod = security_risk_modifier
        selected_emg_mod = emergency_probability_modifier
        selected_emg_impact = emergency_impact

        # 1. Try HDX Data (if available and enabled in model)
        hdx_org_params_used = False
        if hasattr(model, 'hdx_params') and model.hdx_params:
            hdx_security = model.hdx_params.get('security_parameters', {})
            hdx_emergency = model.hdx_params.get('emergency_parameters', {})

            sec_mods = hdx_security.get('security_risk_modifiers', {})
            if org_type in sec_mods and selected_sec_mod is None:
                selected_sec_mod = sec_mods[org_type]
                hdx_org_params_used = True

            emg_mods = hdx_emergency.get('emergency_probability_modifiers', {})
            if org_type in emg_mods and selected_emg_mod is None:
                selected_emg_mod = emg_mods[org_type]
                hdx_org_params_used = True

            emg_impacts = hdx_emergency.get('emergency_impact', {})
            if org_type in emg_impacts and selected_emg_impact is None:
                selected_emg_impact = emg_impacts[org_type]
                hdx_org_params_used = True

            if hdx_org_params_used:
                logger.debug(f"Org {name}: Applied some parameters from HDX data.")

        # 2. Try IATI Data (if available, enabled, and parameter still None)
        iati_org_params = {}
        iati_org_params_used = False
        if hasattr(model, 'iati_params') and model.iati_params:
            iati_org_params = model.iati_params.get('organizations', {}).get(org_type, {})

            if selected_op_rate is None and 'operational_rate' in iati_org_params:
                selected_op_rate = iati_org_params['operational_rate']
                iati_org_params_used = True
            if selected_prem_disc is None and 'premium_discount' in iati_org_params:
                selected_prem_disc = iati_org_params['premium_discount']
                iati_org_params_used = True
            if selected_claim_trig is None and 'claim_trigger' in iati_org_params:
                selected_claim_trig = iati_org_params['claim_trigger']
                iati_org_params_used = True
            if selected_sec_mod is None and 'security_risk_modifier' in iati_org_params:
                selected_sec_mod = iati_org_params['security_risk_modifier']
                iati_org_params_used = True
            if selected_emg_mod is None and 'emergency_probability_modifier' in iati_org_params:
                selected_emg_mod = iati_org_params['emergency_probability_modifier']
                iati_org_params_used = True
            if selected_emg_impact is None and 'emergency_impact' in iati_org_params:
                selected_emg_impact = iati_org_params['emergency_impact']
                iati_org_params_used = True

            if iati_org_params_used:
                logger.debug(f"Org {name}: Applied some parameters from IATI data.")

        # 3. Apply Defaults (if parameter still None)
        default_params = get_iati_parameters_for_model()['organizations'].get(org_type, {})  # Use fallback as source of defaults
        self.operational_rate = selected_op_rate if selected_op_rate is not None else default_params.get('operational_rate', 0.01)  # Default fallback
        
        # Adjust operational rate if too high
        if self.operational_rate > 0.014:
            logger.info(f"Reducing operational rate for {self.name} from {self.operational_rate:.4f} to 0.010")
            self.operational_rate = 0.010  # Reduce operational costs to be sustainable
        
        self.premium_discount = selected_prem_disc if selected_prem_disc is not None else default_params.get('premium_discount', 1.0)
        self.claim_trigger = selected_claim_trig if selected_claim_trig is not None else default_params.get('claim_trigger', 0.5)
        self.security_risk_modifier = selected_sec_mod if selected_sec_mod is not None else default_params.get('security_risk_modifier', 1.0)
        self.emergency_probability_modifier = selected_emg_mod if selected_emg_mod is not None else default_params.get('emergency_probability_modifier', 1.0)
        self.emergency_impact = selected_emg_impact if selected_emg_impact is not None else default_params.get('emergency_impact', 0.1)  # Default fallback

        # Initialize other attributes
        self.insurance_subscribed = False
        self.premium_paid = 0
        self.last_claim_step = -model.waiting_period - 1  # Allow claim in first eligible step

        # Log final selected parameters
        self._log_final_parameters()

    def _log_final_parameters(self):
        """Log final selected parameters for the organization"""
        logger.info(f"Final Parameters for {self.name} ({self.org_type}):")
        logger.info(f"  Operational Rate: {self.operational_rate:.4f}")
        logger.info(f"  Premium Discount: {self.premium_discount:.2f}")
        logger.info(f"  Claim Trigger: {self.claim_trigger:.2f}")
        logger.info(f"  Security Risk Modifier: {self.security_risk_modifier:.2f}")
        logger.info(f"  Emergency Prob Modifier: {self.emergency_probability_modifier:.2f}")
        logger.info(f"  Emergency Impact: {self.emergency_impact:.2f}")

    def receive_funds(self, amount):
        """Receive funds and log the transaction"""
        if amount <= 0:
            return  # Ignore zero/negative funds
        self.balance += amount
        log_msg = f"{self.name} ({self.org_type}) received funds: ${amount:,.0f}. New balance: ${self.balance:,.0f}."
        logger.info(log_msg)
        self.model.log_event(log_msg)

    def subscribe_to_insurance(self):
        """Subscribe to insurance product"""
        if not hasattr(self.model, 'insurance_products') or not self.model.insurance_products:
            logger.warning(f"{self.name}: No insurance products available to subscribe to.")
            return

        try:
            # Use product details for premium calculation (assuming one product for now)
            product = self.model.insurance_products[0]
            # Premium rate is annual, calculate monthly premium
            annual_premium = self.total_budget * (product.get('premium_rate', 1.0) / 100.0)  # Default rate 1%
            monthly_premium = (annual_premium / 12.0) * self.premium_discount  # Apply discount

            if self.balance >= monthly_premium:
                self.balance -= monthly_premium
                self.insurance_subscribed = True
                self.premium_paid += monthly_premium  # Track total premium paid over time
                # Store the premium rate used for payout calc
                self.current_monthly_premium = monthly_premium

                # Update the insurer's records
                if hasattr(self.model, 'insurer'):
                    self.model.insurer.collect_premium(monthly_premium)

                log_msg = f"{self.name} ({self.org_type}) paid monthly premium of ${monthly_premium:,.0f} for {product['name']}. Balance: ${self.balance:,.0f}"
                logger.info(log_msg)
                self.model.log_event(log_msg)
                return True  # Indicate successful subscription/payment
            else:
                log_msg = f"{self.name} ({self.org_type}) cannot afford monthly premium of ${monthly_premium:,.0f} for {product['name']}. Balance: ${self.balance:,.0f}"
                logger.warning(log_msg)
                self.model.log_event(log_msg)
                self.insurance_subscribed = False  # Ensure not subscribed if payment fails
                return False
        except Exception as e:
            logger.error(f"Error in insurance premium payment for {self.name}: {e}", exc_info=True)
            self.insurance_subscribed = False
            return False

    def simulate_claim(self):
        """Simulate insurance claim if conditions met"""
        # Check if eligible to claim
        if not self.insurance_subscribed:
            return
        if self.balance >= self.total_budget * self.claim_trigger:
            return
        if (self.model.current_time - self.last_claim_step) < self.model.waiting_period:
            log_msg = f"{self.name} ({self.org_type}) cannot claim (Step {self.model.current_time}); waiting period ({self.model.waiting_period} steps) not met since last claim (Step {self.last_claim_step})."
            logger.info(log_msg)
            self.model.log_event(log_msg)
            return

        # --- Eligible to Claim ---
        try:
            product = self.model.insurance_products[0]
            # Payout cap based on ANNUAL premium
            annual_premium_estimate = getattr(self, 'current_monthly_premium', 0) * 12
            payout_cap = annual_premium_estimate * product.get('payout_cap', 3.0)  # Default cap 3x

            # Calculate potential payout (how much is needed?)
            amount_needed = (self.total_budget * self.claim_trigger) - self.balance
            # Apply payout cap
            payout = min(amount_needed, payout_cap)

            if payout <= 0:
                logger.info(f"{self.name}: Calculated payout is non-positive (${payout:.0f}), skipping claim.")
                return

            # Process payout via insurer
            if hasattr(self.model, 'insurer'):
                payout_made = self.model.insurer.make_payout(payout)
                if payout_made:
                    self.balance += payout
                    self.last_claim_step = self.model.current_time  # Record claim step

                    log_msg = f"{self.name} ({self.org_type}) made a claim and received payout of ${payout:,.0f}. New balance: ${self.balance:,.0f}."
                    logger.info(log_msg)
                    self.model.log_event(log_msg)

                    # Keep insurance subscribed
                else:
                    # Payout failed (e.g., insurer insolvent)
                    log_msg = f"{self.name} ({self.org_type}) made a claim for ${payout:,.0f}, but insurer could not pay."
                    logger.warning(log_msg)
                    self.model.log_event(log_msg)
            else:
                logger.error(f"{self.name}: Cannot process claim, model has no insurer.")

        except Exception as e:
            logger.error(f"Error in claim simulation for {self.name}: {e}", exc_info=True)

    def step(self):
        """Perform organization's actions for each simulation step (month)"""
        try:
            # 1. Pay Insurance Premium (if applicable)
            # Try to pay premium each month to maintain coverage
            self.subscribe_to_insurance()

            # 2. Deduct Operational Costs with adaptive behavior
            operational_cost = self.total_budget * self.operational_rate
            
            # Adaptive cost reduction when balance is low
            if self.balance < self.total_budget * 0.3:
                # Organizations reduce spending when funds are low
                cost_reduction = 0.35  # Reduce costs by 35% when balance is low
                original_cost = operational_cost
                operational_cost *= (1 - cost_reduction)
                
                log_msg_adapt = f"{self.name} ({self.org_type}) reduced operational costs from ${original_cost:,.0f} to ${operational_cost:,.0f} due to low balance."
                self.model.log_event(log_msg_adapt)
                logger.info(log_msg_adapt)
                
            self.balance -= operational_cost
            log_msg_ops = f"{self.name} ({self.org_type}) spent ${operational_cost:,.0f} on operations."  # Log balance later
            if operational_cost > 0:
                self.model.log_event(log_msg_ops + f" Balance: ${self.balance:,.0f}.")

            # 3. Check for Risk Events
            # Get the current month name (handle time wrap around years)
            month_index = (self.model.current_time % 12) + 1
            current_month_name = datetime.date(1900, month_index, 1).strftime('%b')

            # Apply monthly risk modifiers from HDX data if available
            monthly_security_modifier = 1.0
            if hasattr(self.model, 'hdx_params') and self.model.hdx_params:
                security_params = self.model.hdx_params.get('security_parameters', {})
                monthly_risk_factors = security_params.get('monthly_risk_factors', {})
                monthly_security_modifier = monthly_risk_factors.get(current_month_name, 1.0)
                if monthly_security_modifier != 1.0:
                    logger.debug(f"Org {self.name}: Using monthly security modifier for {current_month_name}: {monthly_security_modifier:.2f}")

            # Check for Emergency Event
            base_emergency_prob = getattr(self.model, 'emergency_probability', 0.035)  # Get from model
            effective_emergency_prob = base_emergency_prob * self.emergency_probability_modifier
            if random.random() < effective_emergency_prob:
                emergency_cost = self.emergency_impact * self.total_budget
                self.balance -= emergency_cost
                log_msg = f"RISK EVENT: {self.name} ({self.org_type}) hit by EMERGENCY (Prob: {effective_emergency_prob:.2%}). Cost: ${emergency_cost:,.0f}. Balance: ${self.balance:,.0f}."
                logger.warning(log_msg)
                # Use detailed event logging
                self.model.log_detailed_event(
                    "Emergency Event",
                    log_msg,
                    self.name,
                    -emergency_cost
                )

            # Check for Security Event
            base_security_risk = getattr(self.model, 'security_risk_factor', 0.12)  # Get from model
            effective_security_risk = base_security_risk * self.security_risk_modifier * monthly_security_modifier
            # Set security impact - higher impact for UN agencies, make configurable?
            security_impact_factor = 0.07 if self.org_type == "UN Agency" else 0.05

            if random.random() < effective_security_risk:
                security_cost = security_impact_factor * self.total_budget
                self.balance -= security_cost
                log_msg = f"RISK EVENT: {self.name} ({self.org_type}) hit by SECURITY incident (Prob: {effective_security_risk:.2%}). Cost: ${security_cost:,.0f}. Balance: ${self.balance:,.0f}."
                logger.warning(log_msg)
                # Use detailed event logging
                self.model.log_detailed_event(
                    "Security Event",
                    log_msg,
                    self.name,
                    -security_cost
                )

            # 4. Simulate Claim (check conditions within the function)
            self.simulate_claim()

            # Log final balance for the step
            logger.debug(f"End of Step {self.model.current_time} for {self.name}: Balance ${self.balance:,.0f}")

        except Exception as e:
            logger.error(f"Error in organization step for {self.name} at step {self.model.current_time}: {e}", exc_info=True)


class InsuranceProviderAgent(Agent):
    """
    Agent representing the insurance provider.
    Collects premiums and makes payouts to organizations.
    """
    def __init__(self, unique_id, model, name="ParaInsure", capital=50000000):
        super().__init__(unique_id, model)
        self.name = name
        self.initial_capital = capital  # Store initial capital
        self.capital = capital
        self.premiums_collected = 0.0
        self.payouts_made = 0.0
        self.profits = 0.0  # Track net profit/loss (Premiums - Payouts)
        logger.info(f"Created Insurance Provider: {self.name} with initial capital ${self.capital:,.0f}")

    def collect_premium(self, amount):
        """Collect premium from an organization and update capital/profits."""
        if amount <= 0:
            return
        self.capital += amount
        self.premiums_collected += amount
        self.profits = self.premiums_collected - self.payouts_made  # Update profit

    def make_payout(self, amount):
        """Pay out an insurance claim and update capital/profits."""
        if amount <= 0:
            return False

        if self.capital >= amount:
            self.capital -= amount
            self.payouts_made += amount
            self.profits = self.premiums_collected - self.payouts_made  # Update profit tracking
            logger.info(f"{self.name} paid out ${amount:,.0f}. Capital remaining: ${self.capital:,.0f}")
            return True  # Payout successful
        else:
            # Handle insolvency - log warning, but maybe still record attempt?
            logger.warning(f"{self.name} is INSOLVENT. Cannot pay out ${amount:,.0f}. Capital: ${self.capital:,.0f}")
            # Should we allow capital to go negative or just fail the payout? Let's fail.
            return False  # Payout failed

class IntegratedBusinessContinuityModel(Model):
    """
    Integrated Business Continuity Insurance Model
    
    This model simulates the financial dynamics of organizations operating in high-risk environments,
    with integrated HDX and IATI data sources for more realistic parameters.
    """
    def __init__(self, sim_duration=12, waiting_period=2, payout_cap_multiple=3.0, 
                 premium_rate=2.0, custom_claim_trigger=None, emergency_probability=None, 
                 security_risk_factor=None, use_hdx_data=True, use_iati_data=True,
                 hdx_data=None, hdx_api_key=None):
        """
        Initialize the integrated business continuity insurance model.
        
        Args:
            sim_duration (int): Simulation duration in months
            waiting_period (int): Minimum months between insurance claims
            payout_cap_multiple (float): Payout cap as multiple of premium
            premium_rate (float): Premium rate as percentage of budget (1.0 = 1%)
            custom_claim_trigger (float, optional): Custom claim trigger threshold
            emergency_probability (float, optional): Override for emergency probability
            security_risk_factor (float, optional): Override for security risk factor
            use_hdx_data (bool): Whether to use HDX data for risk parameters
            use_iati_data (bool): Whether to use IATI data for organization parameters
            hdx_data (dict, optional): Pre-fetched HDX data (e.g., from API)
            hdx_api_key (str, optional): API key for HDX data access
        """
        super().__init__()
        
        # --- Basic Configuration ---
        self.current_time = 0
        self.sim_duration = sim_duration
        self.waiting_period = waiting_period
        self.scheduler = RandomActivation(self)
        self.event_log = []
        self.history = []
        self.profit_history = []
        
        # --- Track data source for reporting ---
        self.hdx_data_source = "Default (No external data)"
        
        # --- Initialize Data Parameters ---
        # First, check if pre-fetched HDX data is provided
        if hdx_data is not None:
            self.hdx_params = hdx_data
            self.hdx_data_source = f"Pre-fetched HDX Data"
            logger.info("Using pre-fetched HDX data.")
        # Otherwise try to fetch data if available and requested
        elif use_hdx_data and HDX_AVAILABLE:
            try:
                # Try to use API key first if provided
                if hdx_api_key:
                    logger.info(f"Attempting to fetch HDX data with API key {hdx_api_key[:4]}...")
                    self.hdx_data_source = f"HDX API (key: {hdx_api_key[:4]}...)"
                    self.hdx_params = fetch_hdx_data(hdx_api_key, country_code="SDN")
                    if self.hdx_params:
                        logger.info("Successfully fetched HDX data with API key.")
                    else:
                        logger.warning("Failed to fetch HDX data with API key. Falling back to cached/simulated data.")
                        self.hdx_data_source = "HDX Cached Data (API fetch failed)"
                
                # If no API key or API fetch failed, use regular cached/simulated data
                if not hdx_api_key or not self.hdx_params:
                    logger.info("Loading HDX security and emergency parameters...")
                    self.hdx_params = get_all_hdx_parameters()
                    if self.hdx_params:
                        logger.info("Successfully loaded HDX parameters.")
                        self.hdx_data_source = "HDX Cached Data"
                    else:
                        logger.warning("Failed to load HDX parameters.")
            except Exception as e:
                logger.error(f"Error loading HDX parameters: {e}", exc_info=True)
                self.hdx_params = None
        else:
            self.hdx_params = None
        
        # Load IATI Data (if available and requested)
        self.iati_params = None
        if use_iati_data and IATI_AVAILABLE:
            try:
                logger.info("Loading IATI organization parameters...")
                self.iati_params = get_iati_parameters_for_model()
                if self.iati_params:
                    logger.info("Successfully loaded IATI parameters.")
                    if self.hdx_data_source == "Default (No external data)":
                        self.hdx_data_source = "IATI Data Only"
                    else:
                        self.hdx_data_source = f"{self.hdx_data_source} + IATI Data"
                else:
                    logger.warning("Failed to load IATI parameters.")
            except Exception as e:
                logger.error(f"Error loading IATI parameters: {e}", exc_info=True)
        
        # --- Set Risk Parameters ---
        # Start with base values (use reduced values)
        self.emergency_probability = 0.035  # 3.5% base probability (reduced from 5%)
        self.security_risk_factor = 0.12    # 12% base probability (reduced from 20%)
        
        # Update from HDX data if available
        if self.hdx_params:
            sec_params = self.hdx_params.get('security_parameters', {})
            emerg_params = self.hdx_params.get('emergency_parameters', {})
            
            # Extract base security risk from HDX if available
            if 'base_security_risk' in sec_params:
                self.security_risk_factor = sec_params['base_security_risk']
                logger.info(f"Using HDX base security risk: {self.security_risk_factor:.2%}")
            
            # Extract emergency probability from HDX if available
            if 'emergency_probability' in emerg_params:
                self.emergency_probability = emerg_params['emergency_probability']
                logger.info(f"Using HDX emergency probability: {self.emergency_probability:.2%}")
        
        # Override with user-provided values if specified
        if emergency_probability is not None:
            self.emergency_probability = emergency_probability
            logger.info(f"Overriding emergency probability with user value: {self.emergency_probability:.2%}")
        
        if security_risk_factor is not None:
            self.security_risk_factor = security_risk_factor
            logger.info(f"Overriding security risk factor with user value: {self.security_risk_factor:.2%}")
        
        # --- Initialize Insurance Products ---
        self.insurance_products = [{
            'name': 'Business Continuity Insurance',
            'premium_rate': premium_rate,          # % of annual budget as premium
            'payout_cap': payout_cap_multiple,     # multiple of premium as maximum payout
            'claim_trigger': custom_claim_trigger  # balance threshold as fraction of budget
        }]
        
        # --- Initialize Insurance Provider ---
        self.insurer = InsuranceProviderAgent(0, self)
        self.scheduler.add(self.insurer)
        
        # --- Initialize Organizations ---
        self.orgs = []
        next_id = 1  # Starting ID for organizations
        
        # Get organization definitions (from IATI if available, otherwise use defaults)
        org_definitions = []
        if use_iati_data and IATI_AVAILABLE:
            try:
                org_definitions = get_organization_definitions_from_iati()
                logger.info(f"Loaded {len(org_definitions)} organization definitions from IATI.")
            except Exception as e:
                logger.error(f"Error getting organization definitions from IATI: {e}", exc_info=True)
        
        # If no IATI defs or error, fall back to default definitions with higher initial capital
        if not org_definitions:
            org_definitions = [
                {"name": "World-Vision", "org_type": "NGO", "initial_capital": 4500000, "total_budget": 8000000},
                {"name": "UNHCR", "org_type": "UN Agency", "initial_capital": 8000000, "total_budget": 15000000},
                {"name": "IOM", "org_type": "Hybrid", "initial_capital": 6000000, "total_budget": 10000000}
            ]
            logger.info("Using default organization definitions with increased initial capital.")
        
        # Create organization agents
        for org_def in org_definitions:
            org = OrgAgent(
                next_id, 
                self,
                name=org_def.get("name", f"Org-{next_id}"),
                org_type=org_def.get("org_type", "NGO"),
                initial_capital=org_def.get("initial_capital", 1000000),
                total_budget=org_def.get("total_budget", 2000000),
                claim_trigger=custom_claim_trigger  # Pass user-specified claim trigger if provided
            )
            self.orgs.append(org)
            self.scheduler.add(org)
            next_id += 1
        
        # --- Initialize Donors ---
        self.donors = []
        
        # Get donor definitions (from IATI if available, otherwise use defaults)
        donor_definitions = []
        if use_iati_data and IATI_AVAILABLE:
            try:
                donor_definitions = get_donor_definitions_from_iati()
                logger.info(f"Loaded {len(donor_definitions)} donor definitions from IATI.")
            except Exception as e:
                logger.error(f"Error getting donor definitions from IATI: {e}", exc_info=True)
        
        # If no IATI defs or error, fall back to default donor definitions
        if not donor_definitions:
            donor_definitions = [
                {"name": "USAID", "donor_type": "11", "avg_donation": 1500000, "donation_frequency": 3,
                 "target_preferences": {"NGO": 0.4, "UN Agency": 0.4, "Hybrid": 0.2}},
                {"name": "ECHO", "donor_type": "40", "avg_donation": 1200000, "donation_frequency": 2.5,
                 "target_preferences": {"NGO": 0.3, "UN Agency": 0.5, "Hybrid": 0.2}},
                {"name": "Private Foundation", "donor_type": "60", "avg_donation": 500000, "donation_frequency": 4,
                 "target_preferences": {"NGO": 0.7, "UN Agency": 0.1, "Hybrid": 0.2}}
            ]
            logger.info("Using default donor definitions.")
        
        # Create donor agents
        for donor_def in donor_definitions:
            donor = DonorAgent(
                next_id,
                self,
                name=donor_def.get("name", f"Donor-{next_id}"),
                donor_type=donor_def.get("donor_type", "10"),
                avg_donation=donor_def.get("avg_donation", 1000000),
                donation_frequency=donor_def.get("donation_frequency", 3),
                target_preferences=donor_def.get("target_preferences", None)
            )
            self.donors.append(donor)
            self.scheduler.add(donor)
            next_id += 1
        
        # Add supplementary donors to improve financial stability
        logger.info("Adding supplementary donors to ensure adequate funding")

        # Create more frequent/smaller donors
        supplementary_donors = [
            {"name": "Regular Small Donors", "donor_type": "60", "avg_donation": 250000, 
             "donation_frequency": 1.5, "target_preferences": {"NGO": 0.5, "UN Agency": 0.2, "Hybrid": 0.3}},
            {"name": "Corporate Sponsor", "donor_type": "70", "avg_donation": 350000,
             "donation_frequency": 2, "target_preferences": {"NGO": 0.3, "UN Agency": 0.3, "Hybrid": 0.4}},
            {"name": "Rapid Response Fund", "donor_type": "40", "avg_donation": 400000,
             "donation_frequency": 3, "target_preferences": {"NGO": 0.4, "UN Agency": 0.4, "Hybrid": 0.2}}
        ]

        for donor_def in supplementary_donors:
            donor = DonorAgent(
                next_id,
                self,
                name=donor_def.get("name", f"Donor-{next_id}"),
                donor_type=donor_def.get("donor_type", "10"),
                avg_donation=donor_def.get("avg_donation", 1000000),
                donation_frequency=donor_def.get("donation_frequency", 3),
                target_preferences=donor_def.get("target_preferences", None)
            )
            self.donors.append(donor)
            self.scheduler.add(donor)
            next_id += 1
        
        # Initialize organization impact history
        self.org_impact_history = {org.name: [] for org in self.orgs}
        
        # Log initialization
        org_count = len(self.orgs)
        donor_count = len(self.donors)
        logger.info(f"Model initialized with {org_count} organizations and {donor_count} donors.")
        logger.info(f"Emergency probability: {self.emergency_probability:.2%}")
        logger.info(f"Security risk factor: {self.security_risk_factor:.2%}")
        logger.info(f"Insurance premium rate: {premium_rate:.2f}%")
        logger.info(f"Insurance payout cap: {payout_cap_multiple:.1f}x premium")
        
        self.log_event(f"=== Model Initialized with {org_count} Organizations and {donor_count} Donors ===")
    
    def log_event(self, message):
        """Add an event to the model's event log with current step"""
        if not hasattr(self, 'event_log'):
            self.event_log = []
        
        step_msg = f"[Step {self.current_time}] " if self.current_time >= 0 else ""
        self.event_log.append(f"{step_msg}{message}")
    
    # In the IntegratedBusinessContinuityModel class, modify the step method

    def step(self):
        """Advance the model by one step (one month)"""
        # Log step start
        self.log_event(f"--- Step {self.current_time} Starting ---")
        
        # Execute all agent steps
        self.scheduler.step()
        
        # Track history for plots
        total_balance = sum(org.balance for org in self.orgs)
        self.history.append(total_balance)
        
        # Track insurer profit history
        insurer_profit = self.insurer.profits if hasattr(self, 'insurer') else 0
        self.profit_history.append(insurer_profit)
        
        # Track individual organization impacts and balance history
        if not hasattr(self, 'org_balance_history'):
            self.org_balance_history = {org.name: [] for org in self.orgs}
        
        for org in self.orgs:
            impact = org.balance - org.initial_capital
            self.org_impact_history[org.name].append(impact)
            
            # Track balance history for each organization
            self.org_balance_history[org.name].append(org.balance)
        
        # Log step end
        self.log_event(f"--- Step {self.current_time} Complete. Total Balance: ${total_balance:,.0f} ---")
        
        # Update current time
        self.current_time += 1

    def log_detailed_event(self, event_type, description, org_name=None, impact=None):
        """Log detailed event information for visualization"""
        if not hasattr(self, 'detailed_events'):
           self.detailed_events = []
        event = {
            'step': self.current_time,
            'type': event_type,
            'description': description,
            'org_name': org_name,
            'impact': impact,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.detailed_events.append(event)
        # Also log to regular event log
        self.log_event(description)
