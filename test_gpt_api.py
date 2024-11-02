from openai import OpenAI
client = OpenAI()


prompt = "Given the article below and the corresponding question, you are expected to choose the correct answer from five candidates to fill @placeholder of abstract concepts in cloze-style machine reading comprehension tasks. Output answer with a single number , ,choose option from [0,1,2,3,4] that best fits the @placeholder in question. article :Media playback is not supported on this device The 21-year-old clocked the fastest times of both the first and second runs and claimed gold by 1.64 seconds. Wendy Holdener, of Switzerland, was second and Sweden's Frida Hansdotter finished third. Great Britain's Alexandra Tilley came in 25th as she finished 5.45 behind the winner in St Moritz on Saturday. Shiffrin's third win in the slalom equalled the record of German Christl Cranz, who completed her treble in 1939., question:American Mikaela Shiffrin won a record - equalling third consecutive world slalom gold with a @placeholder victory at the Alpine World Ski Championships ., 0:dominant, 1:solo,2:impressive,3:thrilling, 4:dramatic, Answer: "
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": prompt,
        }
    ]

    
)

print(completion)