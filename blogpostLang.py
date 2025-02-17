# blog_app.py
import streamlit as st
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
import groq
import re
import os
from dotenv import load_dotenv
load_dotenv()

# Configure Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Configure Groq client
client = groq.Groq(api_key=GROQ_API_KEY)
# Define state structure
class BlogState(TypedDict):
    keyword: str
    titles: List[str]
    selected_title: Optional[str]
    blog_content: Optional[str]

# Initialize LangGraph workflow
def create_workflow():
    workflow = StateGraph(BlogState)

    # Define nodes
    def generate_titles(state: BlogState):
        prompt = f"""Generate 4 blog title options about {state['keyword']}.
        Return ONLY a numbered list following these rules:
        1. Include keyword in first 3 words
        2. Maximum 60 characters
        3. Use power words like 'Essential' or 'Definitive Guide'"""
        
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="qwen-2.5-32b",
                temperature=0.7,
                max_tokens=200
            )
            raw_output = re.sub(r'<\/?[a-zA-Z]+>', '', response.choices[0].message.content, flags=re.DOTALL).strip()
            titles = [line.split(". ", 1)[1].strip() for line in raw_output.split("\n") if ". " in line[:3]][:4]
            return {"titles": titles}
        except Exception as e:
            st.error(f"Title generation failed: {str(e)}")
            return {"titles": []}

    def generate_content(state: BlogState):
        prompt = f"""Write a comprehensive 1500-word blog post titled "{state['selected_title']}".
        Structure with markdown:
        # [Title]
        ## Introduction
        ## Main Content (3-5 sections)
        ### Subsections
        ## Conclusion
        Include practical examples and statistics."""
        
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="qwen-2.5-32b",
                temperature=0.8,
                max_tokens=3000
            )
            content = re.sub(r'<\/?[a-zA-Z]+>', '', response.choices[0].message.content, flags=re.DOTALL).strip()
            return {"blog_content": content}
        except Exception as e:
            st.error(f"Content generation failed: {str(e)}")
            return {"blog_content": ""}

    # Add nodes to workflow
    workflow.add_node("generate_titles", generate_titles)
    workflow.add_node("generate_content", generate_content)

    # Define edges
    workflow.set_entry_point("generate_titles")
    
    def route_after_titles(state: BlogState):
        return "generate_content" if state.get("selected_title") else END

    workflow.add_conditional_edges(
        "generate_titles",
        route_after_titles
    )
    workflow.add_edge("generate_content", END)

    return workflow.compile()

# Streamlit UI
st.title("AI Blog Generator with Qwen-2.5-32b")
app = create_workflow()

# Initialize session state
if 'blog_state' not in st.session_state:
    st.session_state.blog_state = {
        "keyword": "",
        "titles": [],
        "selected_title": None,
        "blog_content": None
    }

# Input Section
keyword = st.text_input("Enter blog topic keyword:", 
                       value=st.session_state.blog_state["keyword"])

# Generate Titles
if st.button("Generate Titles") and keyword.strip():
    with st.spinner("Generating title options..."):
        new_state = app.invoke({
            "keyword": keyword.strip(),
            "titles": [],
            "selected_title": None,
            "blog_content": None
        })
        st.session_state.blog_state = new_state

# Display Titles
if st.session_state.blog_state["titles"]:
    st.subheader("Generated Titles")
    selected_idx = st.radio("Select a title:", 
                           options=range(len(st.session_state.blog_state["titles"])),
                           format_func=lambda x: st.session_state.blog_state["titles"][x])
    
    # Store selection
    st.session_state.blog_state["selected_title"] = st.session_state.blog_state["titles"][selected_idx]

# Generate Content
if st.session_state.blog_state["selected_title"] and not st.session_state.blog_state["blog_content"]:
    if st.button("Generate Full Blog Post"):
        with st.spinner("Writing blog post..."):
            final_state = app.invoke(st.session_state.blog_state)
            st.session_state.blog_state = final_state

# Display Content
if st.session_state.blog_state["blog_content"]:
    st.subheader(st.session_state.blog_state["selected_title"])
    st.markdown(st.session_state.blog_state["blog_content"])
    
    # Download button
    st.download_button(
        "Download Markdown",
        st.session_state.blog_state["blog_content"],
        file_name=f"{st.session_state.blog_state['selected_title']}.md"
    )

# Reset functionality (corrected)
if st.button("Reset"):
    st.session_state.blog_state = {
        "keyword": "",
        "titles": [],
        "selected_title": None,
        "blog_content": None
    }
    st.rerun()  # Using current rerun method