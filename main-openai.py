import streamlit as st 
from utils import print_messages, StreamHandler 
from langchain_core.messages import ChatMessage 
from langchain_core·prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_openai import ChatOpenAI 
import os 

st.set_page_config(page_title="ChatGPT", page_icon= "")   
st.title(" ChatGPT") 

# API KEY 설정 
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"] 

if "messages" not in st.session_state: 
    st.session_state["messages"] = [] 

# 1-2
#채팅 대화기록을 저장하는 store 세션 상태 변수
if "store" not in st.session_state:
    st.session_state["store"] = dict()
# 1-5
with st.sidebar:
    session_id = st.text_input("Session ID", value="abc123")        # 채팅방 아이디 같은 것임. 다른 아이디면 대화가 이어지지 않음!
    
    clear_btn = st.button("대화기록 초기화")
    if clear_btn:
        st.session_state["messages"] = []
        #st.session_state["store"] = dict()     # 아예 대화 저장한 기록까지 초기화.
        st.experimental_rerun()

# 이전 대화기록을 출력해 주는 코드 
print_messages()

# 1-1
#store = {}  # 세션 기록을 저장할 딕셔너리      err: 예)이전의 내용을 영어로 답변해.    (질문 문자 자체를 번역해 버림)

# 1-3
# 세션 ID를 기반으로 세션 기록을 가져오는 함수 
def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    #print(session_ids) 
    #if session_ids not in store:   # 세션 ID가 store에 없는 경우 
    if session_ids not in st.session_state["store"]:    # 세션 ID가 store에 없는 경우 
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장 
        #store[session_ids] = ChatMessageHistory() 
        st.session_state["store"][session_ids] = ChatMessageHistory() 
    #return store[session_ids] # 해당 세션 ID에 대한 세션 기록 반환 
    return st.session_state["store"][session_ids]   # 해당 세션 ID에 대한 세션 기록 반환 


if user_input := st.chat_input("메시지를 입력해 주세요."): 
    # 사용자가 입력한 내용 
    st.chat_message("user").write(f"{user_input}") 
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input)) 
    
    
    # AI의 답변
    with st.chat_message("assistant"):
        # 1-4
        stream_handler = StreamHandler(st.empty())
        
                                                
        # LLM을 사용하여 AI의 답변을 생성 
        #     prompt = ChatPromptTemplate.from_template( 
        #         """질문에 대하여 간결하게 답변해 주세요. 
        # {question} 
        # """
        #     )

        # 1. 모델 생성        
        #llm = ChatOpenAI() 
        # 1-4
        llm = ChatOpenAI(streaming=True, callbacks=[stream_handler]) 
        # 2. 프롬프트 생성
        prompt = ChatPromptTemplate.from_messages( 
            [    
                (                                  
                    "system",
                    "질문에 짧고 간결하게 답변해 주세요.",
                ),
                # 대화 기록을 변수로 사용, history 가 MessageHistory 의 key 가 됨
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),   # 사용자의 질문을 입력
            ]
        )    
        #chain = prompt | ChatOpenAI() | StrOutputParser()
        #chain = prompt | ChatOpenAI() 
        #runnable = prompt | llm
        chain = prompt | llm
        
        chain_with_memory = ( 
            RunnableWithMessageHistory( # RunnableWithMessageHistory 객체 생성 
                chain, # 실행할 Runnable 객체 
                get_session_history, # 세션 기록을 가져오는 함수 
                input_messages_key="question", # 사용자 질문의 키 
                history_messages_key="history", # 기록 메시지의 키 
            )
        )    
            
        # 06-RunnableWithMessageHistory.ipynb
        #response = chain.invoke({"question": user_input})    
        response = chain_with_memory.invoke(         
            #{"question" : "What does cosine mean?"}, 
            {"question" : user_input}, 
            # 세션ID 설정
            config={"configurable": {"session_id": session_id}},
        )
        #msg = chain.invoke({"question": user_input})
        #msg = response.content
        
        
        
        
        # 1-4
        #st.write(msg)
        st.session_state["messages"].append(
            ChatMessage(role="assistant", content=response.content)
        )
        
        
