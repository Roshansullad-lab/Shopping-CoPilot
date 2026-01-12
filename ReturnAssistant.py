from pydantic import BaseModel
import sqlite3, csv, pandas as pd
from datetime import datetime
import os
import json
import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from typing import List, Dict, Any, Optional
from openai import OpenAI
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

DB_FILE = "shopreturns.db"
#CSV_FILE = "C:\\Roshan\\workspace\\sample.csv"
CSV_FILE = "sample.csv"
csvFileReader = None

df = pd.read_csv(CSV_FILE)
print(df.head())

#https://drive.google.com/file/d/1Kn6G89NEfOOejuQmNyS-oXHdQd8g86LH/view?usp=sharing
client = OpenAI(api_key="")


app = FastAPI(title="Shopping copilot")

class InputRequest(BaseModel):
    prompt: str

class ClassResponse(BaseModel):
    role: str
    content: dict

# ----------------------------
# Retrieval Agent
# ----------------------------
class RetrievalAgent:
    def __init__(self, db_file=DB_FILE, csv_file=CSV_FILE):
        self.db_file = db_file
        self.csv_file = csv_file
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_file)
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS returns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT,
            product TEXT,
            category TEXT,
            return_reason TEXT,
            cost TEXT,
            approved_flag TEXT,
            store_name TEXT,
            date TEXT
        )""")
        conn.commit()
        conn.close()

    def ingest_csv(self):
        with open(self.csv_file, newline="") as f:
            return list(csv.DictReader(f))
    
    #search the csv file for the relevant records
    def get_csvfile_context(self,user_text: str) -> str:
        # naive filter: look for product keywords in user_text
        print(user_text)
        print(df['product'])
        print(df)
        

        lower_prompt_input = user_text.lower()
        print(lower_prompt_input)

        items = self.ingest_csv()
        item = next((i for i in items if i["product"].lower() in lower_prompt_input), None)
        if not item:
            print("product missing in the csv file")
            item = next((i for i in items if i["order_id"].lower() in lower_prompt_input), None)
            if not item:
                print("searched using order id that is also not found")
                return None
        print(item)
        matches=json.dumps(item, indent=4) 

        #items = csvFileReader
        #matches = next((i for i in items if i["product"].lower() == user_text), None)
        #if not matches:
        #    print( "product not found in csv file")
        
        #matches = df[user_text.contains(df['product'].astype(str).str, case=False, na=False)]

        #matches = [row for row in df if df['product'] in user_text]
        #if user_text.lower() in  df['product'].str:
        #     matches = df['product']

        #matches = df[df['product'].str.contains(user_text.lower(), case=False, na=False)]
        print("Found record details from csv")
        print(matches)
        # convert to JSON or string for prompt
        return matches
        #return matches.to_json(orient="records")

    def insert_return(self, product: str, reason: str):
        print("Inside insert_return...")
        print(product)
        print(reason)
        items = self.ingest_csv()
        item = next((i for i in items if i["product"].lower() == product.lower()), None)
        if not item:
            return {"status": "error", "message": "Item not found"}
        conn = sqlite3.connect(self.db_file)
        cur = conn.cursor()
#        cur.execute("INSERT INTO returns (order_id,product,reason,date) VALUES (?,?,?,?)",
#                    (item["order_id"], item["product"], reason, datetime.now().isoformat()))
        cur.execute("INSERT INTO returns (order_id,product,category,return_reason,cost,approved_flag,store_name,date) VALUES (?,?,?,?,?,?,?,?)",
                    (item["order_id"],item["product"],item["category"], "dummy",item["cost"],item["approved_flag"],item["store_name"], datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return {"status": "success", "item": item, "reason": reason}

    def handle(self, prompt: str):
        bUseLLM = True
        if True: # "return" in prompt.lower():
            if bUseLLM:
                csv_context = None

                system_prompt = (
                        #"You are a chat assistant that processes customer return requests. "
                        "You are a friendly and efficient AI assistant,that processes customer returns."
                )
                
                messages = [
                            {"role": "system", "content": system_prompt},
                            #{"role": "user", "content": prompt},
                            {"role": "user", "content": prompt},
                         #   {"role": "assistant", "content": f"Customer message: {prompt}"}
                ]

                if "return" in prompt.lower() or "refund" in prompt.lower():
                    system_prompt = (
                        #"You are a chat assistant that processes customer return requests. Respond concisely."
                        "You are a friendly and efficient AI assistant."
                        "Use the CSV context to extract structured fields: "
                        "order_id, product, category, return_reason, cost, approved_flag, store_name, date. "
                        #"Extract structured fields from the message. "
                        #"Fields: order_id, product, category, return_reason, cost, approved_flag, store_name, date. "
                        #"Output valid JSON only."
                    )
                
                    csv_context = self.get_csvfile_context(prompt)
                    #if csv_context is not None:
                    print(csv_context)

                    messages = [
                                {"role": "system", "content": system_prompt},
                                #{"role": "user", "content": prompt},
                                {"role": "user", "content": f"Customer message: {prompt}\nCSV context: {csv_context}"},
                                {"role": "assistant", "content": f"Customer message: {prompt}\nCSV context: {csv_context}"}
                    ]
               
                #messages.append({"role": "user", "content": prompt})

                response = client.chat.completions.create(
                    #model="gpt-5-nano",
                    model="gpt-4o-mini", # "gpt-3.5-turbo" , #"gpt-4",
                    messages=messages,
                    #max_completion_tokens=200
                    #,
                    #temperature=0.2
                )
                content = response.choices[0].message.content
                print(content)
                
                messages.append({"role": "assistant", "content": content})

                if csv_context is None:
                    return {"status": "success", "message": content}

                if "return" in prompt.lower() or "refund" in prompt.lower():
                    try:
                        parsed = json.loads(content)
                        print(parsed)
                        
                    except Exception:
                                parsed = {"error": "Failed to parse JSON", "raw": content}
                                print("exception occured parsing output")
                                #return {"status": "success", "message": "Unable to process return as item not found in our Database."}
                     
                    row_values = json.loads(csv_context)
                    #return self.insert_return(row_values["product"], row_values["return_reason"])
                    return self.insert_return(row_values["product"], parsed.get("return_reason"))
                else:
                    return {"status": "success", "message": content}

            else:
                parts = prompt.split()
                product = parts[1]
                reason = " ".join(parts[2:])
                return self.insert_return(product, reason)
        return {"status": "error", "message": "Unrecognized prompt"}

# ----------------------------
# Report Agent
# ----------------------------
class ReportAgentClass:
    def __init__(self, db_file=DB_FILE):
        self.db_file = db_file

    def get_insights(self):
        conn = sqlite3.connect(self.db_file)
        df = pd.read_sql_query("SELECT * FROM returns", conn)
        conn.close()
        summary = df.groupby("product").size().to_dict() if not df.empty else {}
        json_data = df.to_json(orient="records", date_format="iso")
        return {"summary": summary, "total_returns": len(df), "sql data":json_data}

    def export_excel(self, filename="return_report.xlsx"):
        conn = sqlite3.connect(self.db_file)
        df = pd.read_sql_query("SELECT * FROM returns", conn)
        conn.close()
        if df.empty:
            return {"status": "error", "message": "No data"}
        summary = df.groupby("product").size().reset_index(name="count")
        with pd.ExcelWriter(filename) as writer:
            df.to_excel(writer, sheet_name="Returns", index=False)
            summary.to_excel(writer, sheet_name="Summary", index=False)
        return {"status": "success", "file": filename}

    def handle(self, prompt: str):
        if "report" in prompt.lower():
            self.export_excel()
        return self.get_insights()

# ----------------------------
# Forecasting Agent
# ----------------------------
class ForecastingAgentClass:
    def __init__(self, db_file=DB_FILE):
        self.db_file = db_file

    def forecast_weekly(self):
        conn = sqlite3.connect(self.db_file)
        df = pd.read_sql_query("SELECT * FROM returns", conn)
        conn.close()
        if df.empty:
            return {"forecast": "No data"}
        df["date"] = pd.to_datetime(df["date"])
        weekly = df.groupby(pd.Grouper(key="date", freq="W")).size()
        avg = weekly.mean()
        return {"forecast": f"Expected weekly returns: {avg:.2f}"}

    def forecast_by_category(self):
        conn = sqlite3.connect(self.db_file)
        df = pd.read_sql_query("SELECT * FROM returns", conn)
        conn.close()
        if df.empty:
            return {"forecast": "No data"}
        summary = df.groupby("product").size().to_dict()
        return {"forecast_by_category": summary}

    def handle(self, prompt: str):
        if "weekly" in prompt.lower():
            return self.forecast_weekly()
        elif "category" in prompt.lower():
            return self.forecast_by_category()
        return {"forecast": "Specify daily/weekly or category"}

# ----------------------------
# Coordinator
# ----------------------------
class Coordinator:
    def __init__(self):
        self.retrieval = RetrievalAgent()
        self.report = ReportAgentClass()
        self.forecast = ForecastingAgentClass()

    def route(self, prompt: str):
        if "return" in prompt.lower():
            return ClassResponse(role="retrieval_agent", content=self.retrieval.handle(prompt))
        elif "report" in prompt.lower() or "insight" in prompt.lower():
            return ClassResponse(role="report_agent", content=self.report.handle(prompt))
        elif "forecast" in prompt.lower() or "predict" in prompt.lower():
            return ClassResponse(role="forecast_agent", content=self.forecast.handle(prompt))
        else:
            return ClassResponse(role="retrieval_agent", content=self.retrieval.handle(prompt))
            #return ClassResponse(role="system", content={"status": "error", "message": "Unknown request"})

    async def socketstream(self, prompt: str):
        # Simulate streaming tokens
        text = f"return request for {prompt}."
        #text= coordinator.route(prompt)
        if "report" in prompt.lower() or "insight" in prompt.lower():
            text= self.report.handle(prompt)
        elif "forecast" in prompt.lower() or "predict" in prompt.lower():
            text=self.forecast.handle(prompt)
        else:
            text= self.retrieval.handle(prompt)
        print("sending result")
        print(text)
        #yield text[message]
        strDatakey = "message"
        if "item" in text:
            strDatakey = "item"
        else:
            json_output = json.dumps(text)
            yield json_output
            return

        try:
            json_output = json.dumps(text[strDatakey])
            print(f"Json ouptut to send {text[strDatakey]}")
            #for token in text[strDatakey].split():
            yield json_output
        #    await asyncio.sleep(0.1)  # simulate delay
        except Exception as e:
             print(f"exception occured sending websocket socketstream {e}")



coordinator = Coordinator()

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

#app = FastAPI(title="Shopping copilot")

# ----------------------------
# Serve static files (CSS/JS if needed)
# ----------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

# ----------------------------
# Chat endpoint (your agents logic goes here)
# ----------------------------
class InputRequest(BaseModel):
    prompt: str

class ClassResponse(BaseModel):
    role: str
    content: dict

@app.post("/prompt", response_model=ClassResponse)
def handle_prompt(req: InputRequest):
    return coordinator.route(req.prompt)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        data = await ws.receive_text()
        # Stream response back
        async for token in coordinator.socketstream(data):
            await ws.send_text(token)

#@app.post("/prompt", response_model=ClassResponse)
#def handle_prompt(req: InputRequest):
#    if "return" in req.prompt.lower():
#        return ClassResponse(role="retrieval_agent", content={"message": "Return recorded"})
#    elif "report" in req.prompt.lower():
#        return ClassResponse(role="report_agent", content={"summary": {"Apple TV": 2}, "total_returns": 2})
#    elif "forecast" in req.prompt.lower():
#        return ClassResponse(role="forecast_agent", content={"forecast": "Expected weekly returns: 3.0"})
#    else:
#        return ClassResponse(role="system", content={"message": "Unknown request"})

# ----------------------------
# Serve the HTML UI
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def get_ui():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Shopping copilot</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        #chat { border: 1px solid #ccc; background: #fff; padding: 10px; height: 500px; overflow-y: scroll; border-radius: 8px; }
        .bubble { display: inline-block; padding: 10px; border-radius: 15px; margin: 5px; max-width: 70%; word-wrap: break-word; }
        .user { background-color: #d1e7ff; float: right; clear: both; }
        .assistant { background-color: #e2f7e1; float: left; clear: both; }
        #inputArea { margin-top: 10px; }
        #prompt { width: 70%; padding: 8px; border-radius: 8px; border: 1px solid #ccc; }
        button { padding: 8px 12px; border-radius: 8px; border: none; background: #007bff; color: #fff; cursor: pointer; }
        button:hover { background: #0056b3; }
      </style>
    </head>
    <body>
      <h2>Shopping copilot</h2>
      <div id="chat"></div>
      <div id="inputArea">
        <input id="prompt" type="text" placeholder="Describe your return...">
        <button id="send">Send</button>
      </div>

      <script>
        async function sendPrompt() {
          const prompt = document.getElementById("prompt").value;
          const chatDiv = document.getElementById("chat");

          chatDiv.innerHTML += `<div class="bubble user">You: ${prompt}</div>`;
          chatDiv.scrollTop = chatDiv.scrollHeight;

          const res = await fetch("/prompt", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt })
          });
          const data = await res.json();

          let replyText = "";
          if (data.content.summary) {
            replyText = `Summary: ${JSON.stringify(data.content.summary)}. Total returns: ${data.content.total_returns}.`;
          } else if (data.content.forecast) {
            replyText = `Forecast: ${data.content.forecast}`;
          } else if (data.content.item) {
            replyText = `Return recorded for ${data.content.item.product}. Reason: ${data.content.reason}.`;
          } else if (data.content.message) {
            replyText = data.content.message;
          } else {
            replyText = JSON.stringify(data.content);
          }

          chatDiv.innerHTML += `<div class="bubble assistant">Agent: ${replyText}</div>`;
          document.getElementById("prompt").value = "";
          chatDiv.scrollTop = chatDiv.scrollHeight;
        }
        const chat = document.getElementById('chat');
        const input = document.getElementById('prompt');
        const send = document.getElementById('send');

        function addBubble(text, role) {
          const div = document.createElement('div');
          div.className = 'bubble ' + role;
          div.textContent = text;
          chat.appendChild(div);
          chat.scrollTop = chat.scrollHeight;
          return div;
        }
        function sendData() {
          const message = input.value.trim();
          if (!message) return;
          addBubble(message, 'user');
          input.value = '';

          const ws = new WebSocket(`ws://${location.host}/ws`);
          const assistantBubble = addBubble('', 'assistant');

          ws.onopen = () => ws.send(message);
          ws.onmessage = (event) => {
            assistantBubble.textContent += event.data;
            chat.scrollTop = chat.scrollHeight;
          };
          ws.onclose = () => console.log("WebSocket closed");
        }

        send.onclick = async () => {
          const message = input.value.trim();
          if (!message) return;
          addBubble(message, 'user');
          input.value = '';

          const ws = new WebSocket(`ws://${location.host}/ws`);
          const assistantBubble = addBubble('', 'assistant');

          ws.onopen = () => ws.send(message);
          ws.onmessage = (event) => {
            assistantBubble.textContent += event.data;
            chat.scrollTop = chat.scrollHeight;
          };
          ws.onclose = () => console.log("WebSocket closed");
        };

        input.addEventListener('keydown', function(event) {
          // Check if the ENTER key is pressed AND the SHIFT key is NOT pressed
          if (event.key === 'Enter' && !event.shiftKey) {
            // Prevent the default action (new line in a textarea)
            event.preventDefault();
            // Submit the form
            //form.submit();
            sendData();
          }
        });
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

