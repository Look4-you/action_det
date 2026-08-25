FROM internlm_lmdeploy:v1.5

ENV TZ=Asia/Shanghai
ENV LANG=zh_CN.UTF-8
RUN echo "set encoding=utf-8" > ~/.vimrc

COPY ./model /model

ENV PIP_INDEX_URL=https://artifactory.sf-express.com/artifactory/api/pypi/pip/simple
ENV PIP_TRUSTED_HOST=artifactory.sf-express.com

RUN pip install moviepy ffmpeg-python modelscope
RUN pip install --upgrade transformers==4.41.0

ENV PIP_INDEX_URL=https://artifactory.sit.sf-express.com/artifactory/api/pypi/pip/simple
ENV PIP_TRUSTED_HOST=artifactory.sit.sf-express.com

COPY ./workspace /workspace

WORKDIR /workspace
CMD ["bash", "run-servers.sh"]
