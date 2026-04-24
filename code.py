import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
from langchain.chat_models import init_chat_model

model = init_chat_model("groq:llama-3.3-70b-versatile", temperature=0)
from PyPDF2 import PdfReader

def extract_pdf(file) -> str:
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

file = input("Enter PDF file path: ")
user_text = extract_pdf(file)
from pydantic import BaseModel, Field

class Profitability(BaseModel):
    revenue_growth: str
    gross_margin: str
    net_profit_margin: str
    return_on_equity: str
    return_on_assets: str

class Liquidity(BaseModel):
    current_ratio: str
    quick_ratio: str

class Leverage(BaseModel):
    debt_to_equity: str
    interest_coverage: str

class Efficiency(BaseModel):
    asset_turnover: str
    inventory_turnover: str

class KeyRatios(BaseModel):
    Profitability: Profitability
    Liquidity: Liquidity
    Leverage: Leverage
    Efficiency: Efficiency
    final_observations: list[str]
    conclusion: str

model_with_structure = model.with_structured_output(KeyRatios)

from langchain.messages import SystemMessage, HumanMessage

def ratios_analyser(user_text) -> KeyRatios:
    system_msg = SystemMessage(
    " You are a financial analyser. Extract key ratios from the text and give a detailed, thorough analyisis on the basis of these ratios."
    " Make sure to report only the exact numbers given in the text. If certain numbers or values are not known be clear and explicitely state that."
    " Analyse strengths and weaknesses based on these ratios and also provide a concised conclusion for the same."
    )

    messages = [
    system_msg,
    HumanMessage(user_text)
    ]
    return model_with_structure.invoke(messages)
from typing import Literal

class Flag(BaseModel):
    flag_name: str
    observed_value: str
    explanation: str

class RedFlags(BaseModel):
    # Profitability warnings
    margin_deterioration: bool
    declining_revenue: bool

    # Debt warnings  
    high_debt_levels: bool
    interest_coverage_concern: bool

    # Cash flow warnings
    negative_operating_cashflow: bool 
    cashflow_profit_mismatch: bool   
    
    # Other warnings
    auditor_concerns: bool
    one_time_items: bool  

    # Summary
    flags: list[Flag]                 
    overall_severity: Literal["low", "medium", "high", "critical"]
    risk_summary: str   

model_with_structure2 = model.with_structured_output(RedFlags)

def risk_analyser(user_text) -> RedFlags:
    system_msg2 = SystemMessage(
        "You are a financial risk analyst. "
        "Analyze the financial statement and identify warning signs and risk factors only. "
        "For boolean fields, return True if the risk exists, False if it doesn't. "
        "For the flags list, only include risks that are actually present in the document. "
        "Report only exact values from the document — never estimate or assume. "
        "If a value is not available, state 'not available'."
    )

    messages = [
        system_msg2,
        HumanMessage(user_text)
    ]
    
    return model_with_structure2.invoke(messages)
from langchain_core.runnables import RunnableParallel, RunnableLambda

ratios_runnable = RunnableLambda(lambda text: ratios_analyser(text))
flags_runnable = RunnableLambda(lambda text: risk_analyser(text))

parallel_chain = RunnableParallel({
    "ratios": ratios_runnable,
    "red_flags": flags_runnable
})
result = parallel_chain.invoke(user_text)

class FinancialReport(BaseModel):
    company_name: str
    period: str
    health_score: int
    recommendation: Literal["buy", "hold", "avoid"]
    executive_summary: str

model_with_structure3 = model.with_structured_output(FinancialReport)
def final_analyser(ratios: KeyRatios, red_flags: RedFlags) -> FinancialReport:
    # convert pydantic objects to readable format
    ratios_text = ratios.model_dump()
    flags_text = red_flags.model_dump()

    system_msg3 = SystemMessage(
        "You are a financial analyst."
        "Go through the ratios as well as risk outputs to generate a final financial report."
        "Give advantages and disadvatages based on the ratios and risk analysis."
        "Give a health score which is an integer between 0 and 100"
        "Give an executive summary based on this."
    )
    messages = [
        system_msg3,
        HumanMessage(f"Key Ratios:\n{ratios_text}\n\nRed Flags:\n{flags_text}")
    ]

    return model_with_structure3.invoke(messages)
final_result = final_analyser(result["ratios"], result["red_flags"])
print(final_result)